#!/usr/bin/env python3
"""
Scene MX Modelling – notebook steps as importable Python modules.

This module contains the implementation that was previously located at
`pipeline/components/scene_mx_modelling.py`. The pipeline script remains as a
thin CLI wrapper so container/pipeline paths remain stable.

Key entrypoint:
  - main(): CLI compatible with the wrapper script.
"""

from __future__ import annotations

import os

# Reduce TensorFlow log volume before any TF import (avoids pipe fill when run as subprocess).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from meridian import constants
from meridian.analysis import analyzer
from meridian.analysis import optimizer
from meridian.analysis import summarizer
from meridian.analysis import visualizer
from meridian.data import data_frame_input_data_builder
from meridian.model import model
from meridian.model import prior_distribution
from meridian.model import spec

import argparse
import tempfile
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from google.cloud import storage

import scene_mx
from scene_mx import get_optimized_df, get_response_curves_df, get_summary_df, save_mmm, upload_csv_to_gcs
from scene_mx import utils
from scene_mx import multiprocess_optimization as mpo
from scene_mx.version_check import warn_if_outdated

import tensorflow as tf
import tensorflow_probability as tfp


# -----------------------------------------------------------------------------
# Config & data loading
# -----------------------------------------------------------------------------


def load_config_from_gcs(
    bucket_name: str,
    parameters_path: str,
    config_fname: str,
) -> dict[str, Any]:
    """Load YAML config from GCS and return as dict."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{parameters_path.rstrip('/')}/{config_fname}")
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False) as f:
        blob.download_to_filename(f.name)
        with open(f.name, "r") as rf:
            config = yaml.safe_load(rf)
    return config


def load_input_data(
    bucket_name: str,
    input_data_path: str,
    data_input_file: str,
    sep: str = ",",
) -> pd.DataFrame:
    """Load input CSV/TSV from GCS into a pandas DataFrame."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{input_data_path.rstrip('/')}/{data_input_file}")
    if "tsv" in data_input_file:
        sep = "\t"
    with tempfile.NamedTemporaryFile(delete=False) as f:
        blob.download_to_filename(f.name)
        df = pd.read_csv(f.name, sep=sep)
    return df


def load_model_optional(
    bucket_name: str,
    model_path: str,
    model_fname: str,
):
    """Load a previously trained Meridian model from GCS. Returns the model."""
    return utils.load_pickle_from_gcs(bucket_name, f"{model_path.rstrip('/')}/{model_fname}")


# -----------------------------------------------------------------------------
# Builder: Meridian input data from DataFrame + config
# -----------------------------------------------------------------------------


def build_data(
    df: pd.DataFrame,
    config: dict[str, Any],
):
    """Build Meridian input data using DataFrameInputDataBuilder and config."""
    param_basics = [
        "kpi_type",
        "default_kpi_column",
        "default_time_column",
        "default_media_time_column",
        "default_geo_column",
        "default_population_column",
    ]
    build_params = config.get("build_parameters", {})
    builder_kwargs = {k: build_params[k] for k in param_basics if k in build_params}
    builder = data_frame_input_data_builder.DataFrameInputDataBuilder(**builder_kwargs)

    builder = builder.with_kpi(df)
    if build_params.get("default_population_column") is not None:
        builder = builder.with_population(df)

    var_types = config.get("variable_types", {})
    builder = builder.with_media(
        df,
        media_cols=var_types.get("impression_cols", []),
        media_spend_cols=var_types.get("spend_cols", []),
        media_channels=var_types.get("paid_media_channel", []),
    )
    if var_types.get("controls") is not None:
        builder = builder.with_controls(df, control_cols=var_types["controls"])
    if var_types.get("organic_media_channel") is not None:
        builder = builder.with_organic_media(
            df,
            organic_media_cols=var_types.get("organic_media_cols", []),
            organic_media_channels=var_types["organic_media_channel"],
        )
    if var_types.get("non_media_channel") is not None:
        builder = builder.with_non_media_treatments(
            df,
            non_media_treatment_cols=var_types["non_media_channel"],
        )
    return builder.build()


# -----------------------------------------------------------------------------
# Priors and model spec
# -----------------------------------------------------------------------------


def build_priors(data, config: dict[str, Any]):
    """Build PriorDistribution from config priors aligned to data channels."""
    priors_dict_prev = config.get("priors", {})
    priors_dict_new = {
        k: priors_dict_prev[k]
        for k in data.media_channel.values
        if k in priors_dict_prev
    }
    build_media_channel_args = data.get_paid_media_channels_argument_builder()
    roi_m = build_media_channel_args(
        **{
            channel: tuple(priors_dict_new[channel])
            for channel in data.media_channel.values
        }
    )
    roi_m_mu, roi_m_sigma = zip(*roi_m)
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(
            roi_m_mu, roi_m_sigma, name=constants.ROI_M
        )
    )
    return prior


def build_model_spec(config: dict[str, Any], prior, data, test_pct: float = 0.2, seed: int = 1):
    """Build ModelSpec with holdout_id and config model_spec."""
    np.random.seed(seed)
    tf.random.stateless_uniform([1], seed=[seed, seed])

    n_geos = len(data.geo)
    n_times = len(data.time)
    holdout_id = np.full([n_geos, n_times], False)
    for i in range(n_geos):
        holdout_id[
            i, np.random.choice(n_times, int(np.round(test_pct * n_times)))
        ] = True

    model_spec_args = {
        **config.get("model_spec", {}),
        "prior": prior,
        "holdout_id": holdout_id,
    }
    return spec.ModelSpec(**model_spec_args)


# -----------------------------------------------------------------------------
# Model creation and sampling
# -----------------------------------------------------------------------------


def create_and_sample_model(data, model_spec, config: dict[str, Any]):
    """Create Meridian model and run prior + posterior sampling."""
    mmm = model.Meridian(input_data=data, model_spec=model_spec)
    sampling = config.get("sampling_parameters", {})
    prior_samples = sampling.get("prior_samples", 100)
    mmm.sample_prior(prior_samples)
    mmm.sample_posterior(**sampling.get("posterior_samples", {}))
    return mmm


# -----------------------------------------------------------------------------
# Save model and export parameters
# -----------------------------------------------------------------------------


def save_model_to_gcs(mmm, bucket_name: str, model_path: str, model_fname: Optional[str] = None):
    """Save Meridian model to GCS. If model_fname is None, generates one from date."""
    if model_fname is None:
        current_date = datetime.now().strftime("%Y%m%d")
        model_fname = f"mmm_model_{current_date}.pkl"
    blob_path = f"{model_path.rstrip('/')}/{model_fname}"
    utils.save_mmm_to_gcs(mmm, bucket_name, blob_path)
    return blob_path


def export_parameters_to_yaml_gcs(
    mmm,
    data,
    df: pd.DataFrame,
    config: dict[str, Any],
    bucket_name: str,
    parameters_path: str,
    output_yaml_file: str,
    use_posterior_mean: bool = True,
    sigma_multiplier: float = 0.5,
    assume_log_space: bool = False,
) -> dict[str, Any]:
    """Export parameters and priors to a YAML file and upload to GCS."""
    return utils.export_parameters_to_yaml(
        mmm_model=mmm,
        data=data,
        df=df,
        config_file_loaded=config,
        use_posterior_mean=use_posterior_mean,
        sigma_multiplier=sigma_multiplier,
        assume_log_space=assume_log_space,
        BUCKET_NAME=bucket_name,
        PARAMETERS_PATH=parameters_path,
        output_yaml_file=output_yaml_file,
    )


# -----------------------------------------------------------------------------
# Model analysis and diagnostics
# -----------------------------------------------------------------------------


def prepare_model_diagnostics(mmm) -> pd.DataFrame:
    """Build model diagnostics DataFrame (global + by geo)."""
    model_diagnostics = visualizer.ModelDiagnostics(mmm)
    model_diagnostics_df = model_diagnostics.predictive_accuracy_table(
        selected_geos=None,
        selected_times=None,
        column_var=None,
    )
    model_diagnostics_df["geo"] = "global"

    for region in mmm.input_data.geo.values.tolist():
        md_geo_df = model_diagnostics.predictive_accuracy_table(
            selected_geos=[region],
            selected_times=None,
            column_var=None,
        )
        md_geo_df["geo"] = region
        md_geo_df = md_geo_df[md_geo_df["geo_granularity"] == "geo"]
        model_diagnostics_df = pd.concat(
            [model_diagnostics_df, md_geo_df], ignore_index=True
        )
    return model_diagnostics_df


def prepare_full_summary(mmm, include_baseline: bool = True, adjust_negative_values: bool = False) -> pd.DataFrame:
    """Build full summary DataFrame via scene_mx get_summary_df."""
    model_analysis = analyzer.Analyzer(mmm)
    return get_summary_df(
        model_analysis,
        constants,
        include_baseline=include_baseline,
        adjust_negative_values=adjust_negative_values,
    )


# -----------------------------------------------------------------------------
# Budget optimization
# -----------------------------------------------------------------------------


def _optimization_date_range(mmm, config: dict[str, Any]):
    """Compute optimization period length and (start_date, end_date)."""
    data = mmm.input_data
    opt_params = config.get("optimization_parameters", {})
    raw_weeks = opt_params.get("optimization_period_weeks", "all_time")
    if raw_weeks == "all_time":
        optimization_period_weeks = len(data.time)
    else:
        optimization_period_weeks = raw_weeks
    optimization_start_date = data.time[-optimization_period_weeks:].values.min()
    optimization_end_date = data.time.values.max()
    return optimization_period_weeks, optimization_start_date, optimization_end_date


def _geo_opt_dict(mmm, optimization_start_date, optimization_end_date) -> dict:
    """Build dict of geo -> total_spend for the optimization period."""
    mask = (
        (mmm.input_data.media_spend.time >= str(optimization_start_date))
        & (mmm.input_data.media_spend.time <= str(optimization_end_date))
    )
    total_spend_global = mmm.input_data.media_spend.sel(time=mask).sum().item()
    geo_opt_dict = {"global": {"total_spend": total_spend_global}}
    for geo in mmm.input_data.geo.values.tolist():
        total_spend = (
            mmm.input_data.media_spend.sel(geo=geo).sel(time=mask).sum().item()
        )
        geo_opt_dict[geo] = {"total_spend": total_spend}
    return geo_opt_dict


def _spend_constraints(mmm, config: dict[str, Any]):
    """Build spend_constraint_lower and spend_constraint_upper from config."""
    custom_patterns = config.get("optimization_parameters", {}).get(
        "custom_contsraints_patterns", {}
    )
    expected_channels = list(mmm.input_data.media_channel.values)
    spend_constraint = {}
    for channel in expected_channels:
        channel_str = str(channel)
        constraint_val = 0.3
        if custom_patterns:
            for pattern, constraint_value in custom_patterns.items():
                if pattern in channel_str:
                    constraint_val = constraint_value
                    break
        spend_constraint[channel_str] = constraint_val
    build_channel_args = mmm.input_data.get_paid_channels_argument_builder()
    spend_constraint_lower = build_channel_args(**spend_constraint)
    spend_constraint_upper = build_channel_args(**spend_constraint)
    return spend_constraint_lower, spend_constraint_upper


def prepare_optimization(
    mmm,
    config: dict[str, Any],
    geos_selected: Optional[list] = None,
) -> tuple[pd.DataFrame, dict, Any, Any, str]:
    """Run budget optimizer for all configured geos and scenarios. Returns (optimized_df_clean, geo_opt_dict, optimization_start_date, optimization_end_date, current_date)."""
    budget_optimizer = optimizer.BudgetOptimizer(mmm)
    opt_weeks, optimization_start_date, optimization_end_date = _optimization_date_range(
        mmm, config
    )
    geo_opt_dict = _geo_opt_dict(
        mmm, optimization_start_date, optimization_end_date
    )
    spend_constraint_lower, spend_constraint_upper = _spend_constraints(mmm, config)
    opt_params = config.get("optimization_parameters", {})
    scenarios = opt_params.get("scenarios", [1.0])
    if not isinstance(scenarios, list):
        scenarios = [scenarios]
    batch_size = opt_params.get("batch_size")
    gtol = opt_params.get("gtol")

    geos_to_run = list(geos_selected if geos_selected is not None else geo_opt_dict.keys())
    n_tasks = len(geos_to_run) * len(scenarios)
    if n_tasks > 1:
        print(
            "  In-process: running {} (geo, scenario) tasks sequentially. "
            "For faster runs use the default CLI driver (omit --no_optimization_via_cli) to run tasks in parallel.".format(n_tasks),
            flush=True,
        )

    optimize_kwargs = {}
    if batch_size is not None:
        optimize_kwargs["batch_size"] = int(batch_size)
    if gtol is not None:
        optimize_kwargs["gtol"] = float(gtol)

    optimization_results_fixed_scenarios_dict = {}
    for geo_example in geos_to_run:
        for scenario in scenarios:
            key = f"{geo_example}_{scenario}x"
            print(f"  Optimizing {key} ...", flush=True)
            optimization_results_fixed_scenarios_dict[key] = budget_optimizer.optimize(
                start_date=optimization_start_date,
                end_date=optimization_end_date,
                selected_geos=None if geo_example == "global" else [geo_example],
                fixed_budget=True,
                budget=geo_opt_dict[geo_example]["total_spend"] * scenario,
                spend_constraint_lower=spend_constraint_lower,
                spend_constraint_upper=spend_constraint_upper,
                **optimize_kwargs,
            )

    optimized_scenarios_df = get_optimized_df(optimization_results_fixed_scenarios_dict)
    optimized_scenarios_df["scenario"] = optimized_scenarios_df["geo"].str.extract(
        r"([\d.]+x)$"
    )[0]
    optimized_scenarios_df["geo"] = optimized_scenarios_df["geo"].str.replace(
        r"(_[\d.]+x)$", "", regex=True
    )

    optimized_df_clean = optimized_scenarios_df.copy()
    cols_to_drop = [
        c for c in optimized_df_clean.columns
        if "cpik" in c or "calculated" in c
    ]
    optimized_df_clean = optimized_df_clean.drop(columns=cols_to_drop, errors="ignore")
    current_date = datetime.now().strftime("%Y-%m-%d")
    if hasattr(optimization_start_date, "strftime"):
        optimization_range = (
            optimization_start_date.strftime("%Y-%m-%d")
            + "_"
            + optimization_end_date.strftime("%Y-%m-%d")
        )
    else:
        optimization_range = str(optimization_start_date) + "_" + str(optimization_end_date)
    optimized_df_clean["optimization_date_range"] = optimization_range
    optimized_df_clean["update_date"] = current_date

    return optimized_df_clean, geo_opt_dict, optimization_start_date, optimization_end_date, current_date


def prepare_optimization_via_cli_driver(
    mmm,
    config: dict[str, Any],
    geos_selected: Optional[list] = None,
) -> tuple[pd.DataFrame, dict, Any, Any, str]:
    """Run budget optimization via separate OS processes (CLI driver). Saves model and
    config to temp files, spawns one process per (geo, scenario), then loads parquet
    outputs and returns the same shape as prepare_optimization."""
    opt_weeks, optimization_start_date, optimization_end_date = _optimization_date_range(
        mmm, config
    )
    geo_opt_dict = _geo_opt_dict(
        mmm, optimization_start_date, optimization_end_date
    )
    spend_constraint_lower, spend_constraint_upper = _spend_constraints(mmm, config)
    scenarios = config.get("optimization_parameters", {}).get("scenarios", [1.0])
    opt_params = config.get("optimization_parameters", {})
    if not isinstance(scenarios, list):
        scenarios = [scenarios]
    batch_size = opt_params.get("batch_size")
    if batch_size is not None:
        batch_size = int(batch_size)
    gtol = opt_params.get("gtol")
    if gtol is not None:
        gtol = float(gtol)

    geos_to_run = geos_selected if geos_selected is not None else list(geo_opt_dict.keys())
    geo_opt_dict_subset = {k: geo_opt_dict[k] for k in geos_to_run}

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "mmm.pkl")
        config_path = os.path.join(tmpdir, "opt_config.json")
        local_tmp = os.path.join(tmpdir, "parquet_out")
        os.makedirs(local_tmp, exist_ok=True)

        save_mmm(mmm, model_path)
        mpo.export_config_for_cli(
            geo_opt_dict_subset,
            scenarios,
            optimization_start_date,
            optimization_end_date,
            spend_constraint_lower,
            spend_constraint_upper,
            config_path,
            batch_size=batch_size,
            gtol=gtol,
        )
        mpo.run_driver(model_path, config_path, local_tmp)

        dfs = []
        for geo in geos_to_run:
            for scenario in scenarios:
                key = f"{geo}_{scenario}x"
                parquet_path = os.path.join(local_tmp, f"{key}.parquet")
                if os.path.exists(parquet_path):
                    dfs.append(pd.read_parquet(parquet_path))

    if not dfs:
        raise RuntimeError("CLI driver produced no parquet outputs.")
    optimized_scenarios_df = pd.concat(dfs, ignore_index=True)

    optimized_df_clean = optimized_scenarios_df.copy()
    cols_to_drop = [
        c for c in optimized_df_clean.columns
        if "cpik" in c or "calculated" in c
    ]
    optimized_df_clean = optimized_df_clean.drop(columns=cols_to_drop, errors="ignore")
    current_date = datetime.now().strftime("%Y-%m-%d")
    if hasattr(optimization_start_date, "strftime"):
        optimization_range = (
            optimization_start_date.strftime("%Y-%m-%d")
            + "_"
            + optimization_end_date.strftime("%Y-%m-%d")
        )
    else:
        optimization_range = str(optimization_start_date) + "_" + str(optimization_end_date)
    optimized_df_clean["optimization_date_range"] = optimization_range
    optimized_df_clean["update_date"] = current_date

    return optimized_df_clean, geo_opt_dict, optimization_start_date, optimization_end_date, current_date


# -----------------------------------------------------------------------------
# Response curves
# -----------------------------------------------------------------------------


def prepare_response_curves(
    mmm,
    config: dict[str, Any],
    geos_selected: Optional[list] = None,
    spend_multipliers: Optional[list] = None,
    multiprocess: bool = False,
) -> pd.DataFrame:
    """Build response curves DataFrame. Excludes 'global' from geos if geo_opt_dict provided."""
    model_analysis = analyzer.Analyzer(mmm)
    if geos_selected is not None:
        geos_rc = geos_selected
    else:
        geos_rc = list(mmm.input_data.geo.values)
    if spend_multipliers is None:
        spend_multipliers = list(np.round(np.arange(0, 2.2, 0.1), 1))
    return get_response_curves_df(
        model_analysis,
        geos_selected=geos_rc,
        spend_multipliers=spend_multipliers,
        time_slices_dict=None,
        multiprocess=multiprocess,
    )


# -----------------------------------------------------------------------------
# Save outputs to GCS
# -----------------------------------------------------------------------------


def save_outputs_to_gcs(
    bucket_name: str,
    output_path: str,
    project_name: str,
    current_date: str,
    full_df: Optional[pd.DataFrame] = None,
    response_curves_df: Optional[pd.DataFrame] = None,
    model_diagnostics_df: Optional[pd.DataFrame] = None,
    optimized_df_clean: Optional[pd.DataFrame] = None,
) -> None:
    """Upload MMM summary, response curves, model_eval, and/or optimization to GCS. Only uploads artifacts that are not None."""
    base = output_path.rstrip("/")
    prefix = f"{base}/" if base else ""
    if full_df is not None:
        upload_csv_to_gcs(bucket_name, f"{prefix}mmm_{project_name}_{current_date}", full_df)
    if optimized_df_clean is not None:
        upload_csv_to_gcs(
            bucket_name,
            f"{prefix}optimization_{project_name}_{current_date}",
            optimized_df_clean,
        )
    if response_curves_df is not None:
        upload_csv_to_gcs(
            bucket_name,
            f"{prefix}response_curves_{project_name}_{current_date}",
            response_curves_df,
        )
    if model_diagnostics_df is not None:
        upload_csv_to_gcs(
            bucket_name,
            f"{prefix}model_eval_{project_name}_{current_date}",
            model_diagnostics_df,
        )


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------


def main():
    warn_if_outdated(current_version=scene_mx.__version__)
    parser = argparse.ArgumentParser(
        description="Scene MX Modelling – run notebook steps as modules"
    )
    parser.add_argument(
        "--run_mode",
        choices=["train", "analyze", "optimize"],
        default="train",
        help="train: train and save model, then run analysis (and optionally optimization) + export; "
        "analyze: load existing model (requires --model_fname_prev), run diagnostics/summary/response curves "
        "(and optionally optimization), then export; "
        "optimize: load existing model (requires --model_fname_prev), run only budget optimization and export.",
    )
    parser.add_argument(
        "--skip_optimization",
        action="store_true",
        help="Skip budget optimization: only run prepare_model_diagnostics, prepare_full_summary, "
        "prepare_response_curves and export those (no optimization CSV). Default is to run optimization.",
    )
    parser.add_argument("--bucket_name", required=True, help="GCS bucket name")
    parser.add_argument(
        "--parameters_path",
        default="scene/model/parameters/",
        help="Path to parameters dir in bucket",
    )
    parser.add_argument(
        "--config_fname",
        required=True,
        help="Config YAML filename (e.g. mandarinoriental_parameters_test_2026-02-03.yaml)",
    )
    parser.add_argument(
        "--data_input_file",
        required=True,
        help="Input CSV/TSV filename",
    )
    parser.add_argument(
        "--model_fname_prev",
        default=None,
        help="Previous model pickle filename (required when run_mode=analyze or run_mode=optimize)",
    )
    parser.add_argument(
        "--output_model_fname",
        default=None,
        help="Output model filename when training (default: mmm_model_<date>.pkl)",
    )
    parser.add_argument(
        "--geos_selected",
        nargs="*",
        default=None,
        help="Geos to include in response curves and optimization (e.g. --geos_selected Americas Unknown). Default: all geos.",
    )
    parser.add_argument(
        "--no_optimization_via_cli",
        action="store_true",
        help="Run budget optimization in-process (sequential over geos/scenarios) instead of via CLI driver. "
        "Default: CLI driver runs one process per (geo, scenario) in parallel, which is usually much faster. "
        "When using this flag from a notebook via !python, use python -u or PYTHONUNBUFFERED=1 to avoid stdout deadlock.",
    )
    args = parser.parse_args()
    args.optimization_via_cli = not args.no_optimization_via_cli
    # Empty list from nargs='*' means "all geos"
    geos_list = args.geos_selected if (args.geos_selected and len(args.geos_selected) > 0) else None

    bucket_name = args.bucket_name
    config = load_config_from_gcs(
        bucket_name, args.parameters_path, args.config_fname
    )
    project_params = config.get("project_parameters", {})
    input_data_path = project_params.get("INPUT_DATA_PATH", "scene/data/input/")
    model_path = project_params.get("MODEL_PATH", "scene/model/")
    output_path = project_params.get("OUTPUT_DATA_PATH", "scene/data/output/")
    parameters_path = project_params.get("PARAMETERS_PATH", args.parameters_path)
    client_name = project_params.get("CLIENT_NAME", "scene_mx")
    current_date = datetime.now().strftime("%Y%m%d")

    print("Loading input data...", flush=True)
    df = load_input_data(bucket_name, input_data_path, args.data_input_file)

    if args.run_mode in ("analyze", "optimize"):
        if not args.model_fname_prev:
            raise ValueError(
                "--model_fname_prev required when run_mode=analyze or run_mode=optimize"
            )
        print("Loading pre-trained model...", flush=True)
        mmm = load_model_optional(bucket_name, model_path, args.model_fname_prev)
        data = mmm.input_data
    else:
        print("Building input data...")
        data = build_data(df, config)
        prior = build_priors(data, config)
        model_spec = build_model_spec(config, prior, data)
        print("Creating and sampling model...")
        mmm = create_and_sample_model(data, model_spec, config)
        print("Saving model...")
        save_model_to_gcs(
            mmm, bucket_name, model_path, args.output_model_fname
        )
        print("Exporting parameters to YAML...")
        new_output_yaml_file = f"{client_name}_parameters_{current_date}.yaml"
        export_parameters_to_yaml_gcs(
            mmm, data, df, config,
            bucket_name, parameters_path, new_output_yaml_file,
        )

    if args.run_mode == "optimize":
        # optimize mode: run only budget optimization and export
        if args.optimization_via_cli:
            print("Running budget optimization via CLI driver (separate processes)...", flush=True)
            optimized_df_clean, _, _, _, current_date = prepare_optimization_via_cli_driver(
                mmm, config, geos_selected=geos_list
            )
        else:
            print("Running budget optimization (in-process)...", flush=True)
            optimized_df_clean, _, _, _, current_date = prepare_optimization(
                mmm, config, geos_selected=geos_list
            )
        print("Saving outputs to GCS...", flush=True)
        save_outputs_to_gcs(
            bucket_name,
            output_path,
            client_name,
            current_date,
            optimized_df_clean=optimized_df_clean,
        )
    else:
        # train or analyze: run diagnostics, summary, optional optimization, response curves
        print("Preparing model diagnostics...", flush=True)
        model_diagnostics_df = prepare_model_diagnostics(mmm)
        print("Preparing full summary...", flush=True)
        full_df = prepare_full_summary(mmm)

        # Geo/date range for response curves (exclude 'global' from geo list when present)
        _, optimization_start_date, optimization_end_date = _optimization_date_range(
            mmm, config
        )
        geo_opt_dict = _geo_opt_dict(
            mmm, optimization_start_date, optimization_end_date
        )

        run_optimization = not args.skip_optimization
        optimized_df_clean = None
        if run_optimization:
            if args.optimization_via_cli:
                print("Running budget optimization via CLI driver (separate processes)...", flush=True)
                optimized_df_clean, geo_opt_dict, _, _, current_date = prepare_optimization_via_cli_driver(
                    mmm, config, geos_selected=geos_list
                )
            else:
                print("Running budget optimization (in-process)...", flush=True)
                optimized_df_clean, geo_opt_dict, _, _, current_date = prepare_optimization(
                    mmm, config, geos_selected=geos_list
                )
        else:
            print("Skipping budget optimization (--skip_optimization).", flush=True)

        print("Preparing response curves...", flush=True)
        response_curves_df = prepare_response_curves(
            mmm, config, geos_selected=geos_list
        )
        print("Saving outputs to GCS...", flush=True)
        save_outputs_to_gcs(
            bucket_name,
            output_path,
            client_name,
            current_date,
            full_df=full_df,
            response_curves_df=response_curves_df,
            model_diagnostics_df=model_diagnostics_df,
            optimized_df_clean=optimized_df_clean,
        )
    print("Done.", flush=True)


if __name__ == "__main__":
    main()

