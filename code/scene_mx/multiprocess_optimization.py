"""
Parallel geo-level budget optimization.

Notebook: use run_parallel_optimization(geo_opt_dict=..., mmm=..., ...).

CLI (separate OS processes, no fork/TF issues):
  # First from the notebook: save model and config, then run from shell:
  #   from scene_mx import save_mmm
  #   save_mmm(mmm, "mmm.pkl")
  #   mpo.export_config_for_cli(geo_opt_dict, scenarios, optimization_start_date,
  #       optimization_end_date, spend_constraint_lower, spend_constraint_upper, "config.json")

  python multiprocess_optimization.py --model_path mmm.pkl --config_path config.json --geo APAC --scenario 1.0 --local_tmp ./out &
  python multiprocess_optimization.py --model_path mmm.pkl --config_path config.json --geo Americas --scenario 1.0 --local_tmp ./out &
  wait

  # Or spawn all (geo, scenario) via driver:
  python multiprocess_optimization.py --driver --model_path mmm.pkl --config_path config.json --local_tmp ./out

  # GCS: --model_path and --config_path can be:
  #   - gs://bucket/path (downloaded once in driver)
  #   - bucket_name/blob_path (e.g. mandarinoriental-data-9py6/scene/model/model.pkl) — avoids gs://
  # Keep --local_tmp local; copy results to GCS afterward if needed.
"""
from __future__ import annotations

import os

# Silence TensorFlow/Abseil/XLA log noise (must be set before TF is imported).
# 0=all, 1=no INFO, 2=no INFO/WARNING, 3=no INFO/WARNING/ERROR. Use 2 to hide the
# "absl::InitializeLog" warning and "Compiled cluster using XLA" message.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import multiprocessing as mp
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Union

# pandas is imported only inside run_parallel_optimization (and where needed) so the
# CLI/driver path does not import it; top-level pandas import could cause the driver
# or its child processes to hang when run from a notebook (e.g. pipe buffer / init).

# Set by run_parallel_optimization(); threads and forked children see these.
_worker_mmm = None
_worker_optimizer = None
_worker_get_optimized_df = None
_worker_local_tmp = None


@dataclass
class OptimizationConfig:
    geo_opt_dict: dict
    scenarios: list
    optimization_start_date: Union[str, "pandas.Timestamp"]
    optimization_end_date: Union[str, "pandas.Timestamp"]
    spend_constraint_lower: float
    spend_constraint_upper: float
    local_tmp: Union[Path, str, None]  # passed in task so workers get it under spawn


def run_one_optimization(args):
    # Allow (geo, scenario, cfg, budget_optimizer) when reuse_optimizer=True (sequential)
    if len(args) == 4:
        geo, scenario, cfg, budget_optimizer = args
    else:
        geo, scenario, cfg = args
        budget_optimizer = _worker_optimizer.BudgetOptimizer(_worker_mmm)
    key = f"{geo}_{scenario}x"
    print(f"[Task] Starting {key} ...", flush=True)
    out_dir = Path(cfg.local_tmp) if cfg.local_tmp is not None else Path("./optimization_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.parquet"

    if out_path.exists():
        if len(args) == 4:
            # Sequential: return (key, None) so caller can load from parquet if needed
            return (key, None)
        return f"Skipped {key}"

    print(f"[Task] Optimizing {key} ({scenario}x historic spend) ...", flush=True)
    result = budget_optimizer.optimize(
        start_date=cfg.optimization_start_date,
        end_date=cfg.optimization_end_date,
        selected_geos=None if geo == "global" else [geo],
        fixed_budget=True,
        budget=cfg.geo_opt_dict[geo]["total_spend"] * scenario,
        spend_constraint_lower=cfg.spend_constraint_lower,
        spend_constraint_upper=cfg.spend_constraint_upper,
    )

    # Transform via get_optimized_df (scene_mx) for parquet and for returned dfs
    df = _worker_get_optimized_df(result)
    df["geo"] = geo
    df["scenario"] = f"{scenario}x"
    df.to_parquet(
        out_path,
        engine="pyarrow",
        compression="snappy",
    )
    print(f"[Task] Done {key}", flush=True)
    if len(args) == 4:
        return (key, df)
    return f"Done {key}"


def run_parallel_optimization(
    geo_opt_dict: dict,
    scenarios: list,
    mmm,
    optimization_start_date,
    optimization_end_date,
    spend_constraint_lower: float,
    spend_constraint_upper: float,
    optimizer_module,
    get_optimized_df_fn: Callable,
    local_tmp=None,
    n_processes: int = 2,
    use_multiprocessing: bool = False,
    use_threading: bool = False,
    return_dfs: bool = False,
):
    """
    Run geo optimizations (sequential, threaded, or multiprocess). Call from
    the notebook with your variables (geo_opt_dict, mmm, optimizer, etc.).

    Parquet files are always written (after get_optimized_df() transformation).
    Set return_dfs=True only when you need the DataFrames in memory (sequential
    only); otherwise False saves memory (default).

    Parallel options (no fork):
      use_threading=True  — Use threads instead of processes. Same memory as
        notebook (mmm, optimizer); TF often releases the GIL during heavy ops
        so you can get real parallelism without fork/TF issues. Prefer over
        use_multiprocessing when running in the notebook.
      use_multiprocessing=True — Uses fork; can hang with TensorFlow (not
        fork-safe). Avoid unless you know it works in your environment.

    Returns:
        When sequential and return_dfs=True: dict[str, DataFrame]. Otherwise:
        None (load from local_tmp parquet when needed). With use_threading,
        return_dfs is ignored and None is returned.

    Example (parallel via threading, no fork/TF issues):

        mpo.run_parallel_optimization(
            ...,
            n_processes=4,
            use_threading=True,
        )
    """
    global _worker_mmm, _worker_optimizer, _worker_get_optimized_df, _worker_local_tmp
    _worker_mmm = mmm
    _worker_optimizer = optimizer_module
    _worker_get_optimized_df = get_optimized_df_fn
    _worker_local_tmp = Path(local_tmp) if local_tmp is not None else Path("./optimization_results")
    _worker_local_tmp.mkdir(parents=True, exist_ok=True)

    resolved_local_tmp = Path(local_tmp) if local_tmp is not None else Path("./optimization_results")
    resolved_local_tmp.mkdir(parents=True, exist_ok=True)

    cfg = OptimizationConfig(
        geo_opt_dict=geo_opt_dict,
        scenarios=scenarios,
        optimization_start_date=optimization_start_date,
        optimization_end_date=optimization_end_date,
        spend_constraint_lower=spend_constraint_lower,
        spend_constraint_upper=spend_constraint_upper,
        local_tmp=str(resolved_local_tmp),  # str so it pickles reliably to workers
    )

    tasks = [
        (geo, scenario, cfg)
        for geo in cfg.geo_opt_dict.keys()
        for scenario in cfg.scenarios
    ]

    if use_threading:
        # Threads share process memory: no fork, no TF copy. TF releases GIL
        # during heavy ops so threading can give real parallelism.
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            for msg in executor.map(run_one_optimization, tasks):
                print(msg, flush=True)
        return None  # parquet only; return_dfs not supported with threading
    elif use_multiprocessing:
        # Fork: workers inherit _worker_* globals. TensorFlow is not fork-safe
        # and can hang in forked workers; prefer use_threading or sequential.
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n_processes) as pool:
            for msg in pool.imap_unordered(run_one_optimization, tasks):
                print(msg, flush=True)
        return None  # parquet only; load from local_tmp if you need dfs
    else:
        # Sequential: reuse one BudgetOptimizer; parquet always written via get_optimized_df().
        budget_optimizer = optimizer_module.BudgetOptimizer(mmm)
        results_dfs = {} if return_dfs else None
        if return_dfs:
            import pandas as pd
        for geo, scenario, cfg in ((t[0], t[1], t[2]) for t in tasks):
            ret = run_one_optimization((geo, scenario, cfg, budget_optimizer))
            key = ret[0]
            df = ret[1]
            if df is not None:
                if return_dfs:
                    results_dfs[key] = df
                print(f"Done {key}", flush=True)
                continue
            # Skipped (already on disk)
            if return_dfs:
                out_path = Path(cfg.local_tmp) / f"{key}.parquet"
                if out_path.exists():
                    results_dfs[key] = pd.read_parquet(out_path)
            print(f"Skipped {key}", flush=True)
        return results_dfs if return_dfs else None


def _json_serializer(obj):
    """Convert numpy/pandas types for JSON."""
    import numpy as np
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_config_for_cli(
    geo_opt_dict: dict,
    scenarios: list,
    optimization_start_date: Union[str, "pandas.Timestamp"],
    optimization_end_date: Union[str, "pandas.Timestamp"],
    spend_constraint_lower: Union[float, dict],
    spend_constraint_upper: Union[float, dict],
    config_path: str,
    batch_size: Optional[int] = None,
    gtol: Optional[float] = None,
):
    """Save config to JSON for use with CLI/process-level parallelism. Call from notebook.
    spend_constraint_lower/upper can be a scalar float or per-channel dict (from _spend_constraints)."""
    import json
    start_val = optimization_start_date
    end_val = optimization_end_date
    start_str = start_val.isoformat() if hasattr(start_val, "isoformat") else str(start_val)
    end_str = end_val.isoformat() if hasattr(end_val, "isoformat") else str(end_val)
    data = {
        "geo_opt_dict": geo_opt_dict,
        "scenarios": scenarios,
        "optimization_start_date": start_str,
        "optimization_end_date": end_str,
        "spend_constraint_lower": spend_constraint_lower,
        "spend_constraint_upper": spend_constraint_upper,
    }
    if batch_size is not None:
        data["batch_size"] = batch_size
    if gtol is not None:
        data["gtol"] = gtol
    with open(config_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)
    print(f"Config written to {config_path}", flush=True)


def _download_from_gcs(gs_uri: str, dest_path: str) -> str:
    """Download gs://bucket/path to dest_path. Returns dest_path."""
    uri = (gs_uri or "").strip()
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got {uri!r}")
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "GCS URIs require google-cloud-storage. Install with: pip install google-cloud-storage"
        ) from None
    parts = uri[5:].split("/", 1)  # skip "gs://"
    bucket_name, blob_path = parts[0], parts[1]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(dest_path)
    return dest_path


def _resolve_to_local(uri_or_path: str, tmpdir: Optional[str] = None) -> str:
    """If uri_or_path is gs://..., download to a temp file and return local path. Else return resolved absolute path."""
    uri = (uri_or_path or "").strip()
    if not uri.startswith("gs://"):
        return str(Path(uri).resolve())
    suffix = Path(uri).name or "file"
    if tmpdir:
        os.makedirs(tmpdir, exist_ok=True)
        local = os.path.join(tmpdir, suffix)
    else:
        fd, local = tempfile.mkstemp(suffix="_" + suffix)
        os.close(fd)
    _download_from_gcs(uri, local)
    return local


def _normalize_path_arg(uri_or_path: str) -> str:
    """Strip whitespace and BOM so GCS detection works across environments (e.g. GCP shell)."""
    s = (uri_or_path or "").strip()
    if s.startswith("\ufeff"):
        s = s[1:].strip()
    return s


def _is_gcs_uri(uri_or_path: str) -> bool:
    """True if the string is a GCS URI (after normalize)."""
    return _normalize_path_arg(uri_or_path).startswith("gs://")


def _parse_gcs_bucket_blob(uri_or_path: str) -> Optional[tuple[str, str]]:
    """If path looks like 'bucket_name/blob_path' (no gs://, no leading /), return (bucket, blob). Else None."""
    s = _normalize_path_arg(uri_or_path)
    if not s or s.startswith("gs://") or s.startswith("/") or "/" not in s:
        return None
    bucket, _, blob = s.partition("/")
    if not bucket or not blob:
        return None
    return (bucket, blob)


def _download_gcs_blob_to_file(bucket_name: str, blob_path: str, dest_path: str) -> str:
    """Download a GCS blob to a local file. Returns dest_path."""
    try:
        from google.cloud import storage
    except ImportError:
        raise ImportError(
            "GCS requires google-cloud-storage. Install with: pip install google-cloud-storage"
        ) from None
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(dest_path)
    return dest_path


def _load_model_from_path(path: Path, model_path: str):
    """Load MMM from a local file path."""
    size = path.stat().st_size
    if size == 0:
        raise ValueError(
            f"Model file is empty (0 bytes): {path}. "
            "Re-save the model from the notebook (e.g. save_mmm(mmm, 'path.pkl')) or check the path."
        )
    if size < 100:
        raise ValueError(
            f"Model file is too small ({size} bytes) to be valid: {path}. "
            "File may be truncated or not a model pickle."
        )
    try:
        from meridian.model import model
        return model.load_mmm(str(path))
    except EOFError as e:
        raise ValueError(
            f"Model file is truncated or corrupted (EOFError): {path}. "
            f"File size: {size} bytes. Re-save the model (save_mmm(mmm, ...)) or re-upload."
        ) from e
    except Exception:
        import joblib
        try:
            with open(path, "rb") as f:
                return joblib.load(f)
        except EOFError as e:
            raise ValueError(
                f"Model file is truncated or corrupted (EOFError): {path}. File size: {size} bytes."
            ) from e


def _load_model(model_path: str):
    """Load MMM from disk, gs:// URI, or GCS bucket/blob (uses utils.load_pickle_from_gcs for blob)."""
    normalized_path = _normalize_path_arg(model_path)
    path = Path(normalized_path).resolve()

    # 1) gs:// URI → download to temp then load from file
    if _is_gcs_uri(normalized_path):
        local_path = Path(_resolve_to_local(normalized_path))
        return _load_model_from_path(local_path, normalized_path)

    # 2) bucket/blob form (e.g. "bucket_name/path/to/model.pkl") when not a local file
    if not path.exists():
        gcs = _parse_gcs_bucket_blob(normalized_path)
        if gcs:
            bucket_name, blob_path = gcs
            from scene_mx import utils
            return utils.load_pickle_from_gcs(bucket_name, blob_path)
        raise FileNotFoundError(
            f"Model path not found: {normalized_path}. "
            "Use a local path, gs://bucket/path, or bucket_name/blob_path."
        )

    return _load_model_from_path(path, normalized_path)


def _load_cli_config(config_path: str):
    """Load config JSON from local path, gs:// URI, or GCS bucket/blob."""
    import json
    path_or_uri = _normalize_path_arg(config_path)
    if _is_gcs_uri(path_or_uri):
        local_path = _resolve_to_local(path_or_uri)
        with open(local_path) as f:
            return json.load(f)
    if not Path(path_or_uri).resolve().exists():
        gcs = _parse_gcs_bucket_blob(path_or_uri)
        if gcs:
            bucket_name, blob_path = gcs
            fd, local = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            _download_gcs_blob_to_file(bucket_name, blob_path, local)
            with open(local) as rf:
                return json.load(rf)
        raise FileNotFoundError(f"Config path not found: {path_or_uri}")
    with open(path_or_uri) as f:
        return json.load(f)


def run_single_task_cli(
    model_path: str,
    config_path: str,
    geo: str,
    scenario: float,
    local_tmp: str = "./optimization_results",
):
    """
    Run one (geo, scenario) optimization. Used by CLI and by driver subprocesses.
    Loads model and config from disk, runs optimization, writes parquet.
    """
    mmm = _load_model(model_path)
    config = _load_cli_config(config_path)
    geo_opt_dict = config["geo_opt_dict"]
    if geo not in geo_opt_dict:
        raise ValueError(f"Unknown geo {geo!r}; keys: {list(geo_opt_dict.keys())}")

    from meridian.analysis import optimizer
    from scene_mx import get_optimized_df

    budget_optimizer = optimizer.BudgetOptimizer(mmm)
    out_dir = Path(local_tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = f"{geo}_{scenario}x"
    out_path = out_dir / f"{key}.parquet"
    if out_path.exists():
        print(f"Skipped {key}", flush=True)
        return 0

    start = config["optimization_start_date"]
    end = config["optimization_end_date"]
    lo = config["spend_constraint_lower"]
    hi = config["spend_constraint_upper"]
    budget = geo_opt_dict[geo]["total_spend"] * scenario

    optimize_kwargs = {}
    if config.get("batch_size") is not None:
        optimize_kwargs["batch_size"] = int(config["batch_size"])
    if config.get("gtol") is not None:
        optimize_kwargs["gtol"] = float(config["gtol"])

    print(f"Optimizing {key} ...", flush=True)
    result = budget_optimizer.optimize(
        start_date=start,
        end_date=end,
        selected_geos=None if geo == "global" else [geo],
        fixed_budget=True,
        budget=budget,
        spend_constraint_lower=lo,
        spend_constraint_upper=hi,
        **optimize_kwargs,
    )
    df = get_optimized_df(result)
    df["geo"] = geo
    df["scenario"] = f"{scenario}x"
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    print(f"Done {key}", flush=True)
    return 0


def run_driver(
    model_path: str,
    config_path: str,
    local_tmp: str,
    max_concurrent: int = 4,
):
    """Spawn OS processes for (geo, scenario) tasks with bounded concurrency.

    Args:
        model_path: Path/URI to the MMM model (local, gs://, or bucket/blob).
        config_path: Path/URI to the optimization config JSON.
        local_tmp: Local directory for parquet outputs.
        max_concurrent: Maximum number of child processes to run at once.
            Default 4. Use a small value (e.g. 2–4) on 4‑CPU machines.
    """
    import subprocess
    import sys
    import threading

    if max_concurrent < 1:
        raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

    def _read_and_print(pipe, prefix: str):
        """Read lines from pipe and print (avoids pipe buffer deadlock when many children share stdout)."""
        try:
            for line in pipe:
                print(f"{prefix}{line}", end="", flush=True)
        except (BrokenPipeError, OSError) as e:
            # Expected when child exits or pipe is closed; no action needed.
            _ = e
        finally:
            try:
                pipe.close()
            except OSError as e:
                _ = e  # ignore on close

    # GCS: download model/config once to a temp dir so all children use local paths (avoids N downloads)
    resolved_model_path = _normalize_path_arg(model_path)
    resolved_config_path = _normalize_path_arg(config_path)
    need_cleanup = False
    tmpdir = None
    if (
        _is_gcs_uri(resolved_model_path)
        or _is_gcs_uri(resolved_config_path)
        or _parse_gcs_bucket_blob(resolved_model_path)
        or _parse_gcs_bucket_blob(resolved_config_path)
    ):
        tmpdir = tempfile.mkdtemp(prefix="mpo_driver_")
        need_cleanup = True
        if _is_gcs_uri(resolved_model_path):
            resolved_model_path = _resolve_to_local(resolved_model_path, tmpdir=tmpdir)
        elif _parse_gcs_bucket_blob(resolved_model_path):
            bucket, blob = _parse_gcs_bucket_blob(resolved_model_path)
            resolved_model_path = os.path.join(tmpdir, Path(blob).name or "model.pkl")
            _download_gcs_blob_to_file(bucket, blob, resolved_model_path)
        else:
            resolved_model_path = str(Path(resolved_model_path).resolve())
        if _is_gcs_uri(resolved_config_path):
            resolved_config_path = _resolve_to_local(resolved_config_path, tmpdir=tmpdir)
        elif _parse_gcs_bucket_blob(resolved_config_path):
            bucket, blob = _parse_gcs_bucket_blob(resolved_config_path)
            resolved_config_path = os.path.join(tmpdir, Path(blob).name or "config.json")
            _download_gcs_blob_to_file(bucket, blob, resolved_config_path)
        else:
            resolved_config_path = str(Path(resolved_config_path).resolve())
    else:
        resolved_model_path = str(Path(resolved_model_path).resolve())
        resolved_config_path = str(Path(resolved_config_path).resolve())
    resolved_local_tmp = str(Path(local_tmp).resolve())
    config = _load_cli_config(resolved_config_path)
    geo_opt_dict = config["geo_opt_dict"]
    scenarios = config["scenarios"]
    script = Path(__file__).resolve()
    # Build list of all (geo, scenario) tasks up front.
    tasks = [(geo, scenario) for geo in geo_opt_dict for scenario in scenarios]

    procs = []   # list[(key, Popen)]
    readers = [] # list[Thread]

    def _start_task(geo: str, scenario) -> None:
        key = f"{geo}_{scenario}x"
        cmd = [
            sys.executable,
            str(script),
            "--model_path", resolved_model_path,
            "--config_path", resolved_config_path,
            "--geo", geo,
            "--scenario", str(scenario),
            "--local_tmp", resolved_local_tmp,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append((key, p))
        t = threading.Thread(target=_read_and_print, args=(p.stdout, f"[{key}] "))
        t.daemon = True
        t.start()
        readers.append(t)

    # Enforce a simple concurrency cap: keep at most max_concurrent children running.
    for geo, scenario in tasks:
        # If we are at the concurrency limit, block until at least one running child finishes.
        while len([p for _k, p in procs if p.poll() is None]) >= max_concurrent:
            for _key, p in procs:
                if p.poll() is None:
                    p.wait()
                    break
        _start_task(geo, scenario)

    # Wait for all children to complete.
    for _key, p in procs:
        p.wait()
    for t in readers:
        t.join(timeout=1.0)
    failed = [key for key, p in procs if p.returncode != 0]
    if need_cleanup and tmpdir:
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to remove temporary directory {tmpdir}: {e}", stacklevel=0)
    if failed:
        sys.exit(1)
    print("All tasks finished.", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run geo optimization: one (geo, scenario) per process. "
        "From notebook use run_parallel_optimization() instead."
    )
    parser.add_argument(
        "--model_path",
        help="Path to MMM: local path, gs://bucket/path, or bucket_name/blob_path (GCS)",
    )
    parser.add_argument(
        "--config_path",
        help="Path to config JSON: local path, gs://bucket/path, or bucket_name/blob_path (GCS)",
    )
    parser.add_argument("--geo", help="Geo to optimize (e.g. APAC, Americas, global)")
    parser.add_argument("--scenario", type=float, help="Scenario multiplier (e.g. 1.0, 1.15)")
    parser.add_argument("--local_tmp", default="./optimization_results", help="Output directory for parquet files")
    parser.add_argument("--driver", action="store_true", help="Spawn one process per (geo, scenario); no --geo/--scenario needed")
    args = parser.parse_args()

    if args.driver:
        if not args.model_path or not args.config_path:
            parser.error("--driver requires --model_path and --config_path")
        run_driver(args.model_path, args.config_path, args.local_tmp or "./optimization_results")
        return
    if not all([args.model_path, args.config_path, args.geo, args.scenario is not None]):
        parser.error("Single-task run requires --model_path, --config_path, --geo, --scenario")
    run_single_task_cli(
        model_path=args.model_path,
        config_path=args.config_path,
        geo=args.geo,
        scenario=args.scenario,
        local_tmp=args.local_tmp or "./optimization_results",
    )


if __name__ == "__main__":
    main()