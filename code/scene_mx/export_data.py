"""Module for exporting optimization results to pandas DataFrames.

This module provides functions to convert optimization results from the Meridian
optimizer into pandas DataFrames for easier analysis and visualization.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np
import xarray as xr
import os
from datetime import datetime
from meridian.analysis.optimizer import OptimizationResults
from meridian.analysis import analyzer
from meridian import constants
import joblib  # <-- Fix NameError by importing joblib

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Google Cloud Storage functionality is not available.")
    print("To enable GCS uploads, install google-cloud-storage using:")
    print("pip install google-cloud-storage")

def dataset_to_dataframe(ds: xr.Dataset) -> pd.DataFrame:
    """
    Converts an xarray Dataset to a pandas DataFrame.

    This function attempts to create a DataFrame by:
    1. Resetting all multi-indices in the Dataset to become regular columns.
    2. Combining all data variables into the DataFrame.

    Args:
        ds (xr.Dataset): The xarray Dataset to convert.

    Returns:
        pd.DataFrame: The resulting pandas DataFrame.
    """
    # Reset all multi-indices to become columns
    ds_reset = ds.reset_coords(drop=False)

    # Convert to DataFrame
    df = ds_reset.to_dataframe().reset_index()

    return df


def get_optimized_df(
    optimization_results: Union[OptimizationResults, Dict[str, OptimizationResults]],
    adjust_negative_values: bool = True
) -> pd.DataFrame:
    """Convert optimization results to a pandas DataFrame.

    This function takes optimization results from either global or geo-specific
    optimization and converts them into a pandas DataFrame for easier analysis.
    It handles both single-geo and multi-geo optimization results.

    Args:
        optimization_results: Either a single OptimizationResults object for global
            optimization, or a dictionary mapping geo names to OptimizationResults
            objects for geo-specific optimization.
        adjust_negative_values: If True, replaces negative and infinite values
            with 0 in the output DataFrame. Defaults to True.

    Returns:
        A pandas DataFrame containing the optimization results with the following
        columns:
        - geo: Geography identifier ('global' for global optimization)
        - channel: Media channel name
        - incremental_outcome: Optimized incremental outcome
        - pct_of_contribution: Percentage contribution to total outcome
        - spend: Optimized spend
        - roi: Return on investment
        - roi_calculated: Calculated ROI (incremental_outcome/spend)
        - mroi: Marginal return on investment
        - cpik: Cost per incremental KPI
        - pct_of_spend: Percentage of total spend
        - effectiveness: Channel effectiveness
        - *_nonoptimized: Corresponding metrics for non-optimized scenario

    Example:
        >>> # For global optimization
        >>> df = get_optimized_df(global_optimization_results)
        >>> # For geo-specific optimization
        >>> df = get_optimized_df(geo_optimization_results)
    """
    optimized_df_global = pd.DataFrame()

    if hasattr(optimization_results, 'keys'):
        # geo optimization output
        for geo in optimization_results.keys():

            opt_data = optimization_results[geo].optimized_data
            nonopt_data = optimization_results[geo].nonoptimized_data
            if 'metric' in opt_data.dims:
                opt_data = opt_data.sel(metric='mean')
            if 'metric' in nonopt_data.dims:
                nonopt_data = nonopt_data.sel(metric='mean')

            if hasattr(opt_data, 'pct_of_contribution') and hasattr(nonopt_data, 'pct_of_contribution'):
                pct_of_contribution_optimized = opt_data.pct_of_contribution.values
                pct_of_contribution_nonoptimized = nonopt_data.pct_of_contribution.values
            else:
                total_outcome_optimized = np.sum(opt_data.incremental_outcome.values)
                pct_of_contribution_optimized = opt_data.incremental_outcome.values / total_outcome_optimized if total_outcome_optimized > 0 else np.zeros_like(opt_data.incremental_outcome.values)
                total_outcome_nonoptimized = np.sum(nonopt_data.incremental_outcome.values)
                pct_of_contribution_nonoptimized = nonopt_data.incremental_outcome.values / total_outcome_nonoptimized if total_outcome_nonoptimized > 0 else np.zeros_like(nonopt_data.incremental_outcome.values)

            optimized_df_ = pd.DataFrame({
                'geo': geo,
                'channel': opt_data.channel.values,
                'incremental_outcome': opt_data.incremental_outcome.values,
                'pct_of_contribution': pct_of_contribution_optimized,
                'spend': opt_data.spend.values,
                'roi': opt_data.roi.values,
                'roi_calculated': opt_data.incremental_outcome.values / opt_data.spend.values,
                'mroi': opt_data.mroi.values,
                'cpik': opt_data.cpik.values,
                'pct_of_spend': opt_data.pct_of_spend.values,
                'incremental_outcome_nonoptimized': nonopt_data.incremental_outcome.values,
                'pct_of_contribution_nonoptimized': pct_of_contribution_nonoptimized,
                'spend_nonoptimized': nonopt_data.spend.values,
                'roi_nonoptimized': nonopt_data.roi.values,
                'roi_calculated_nonoptimized': nonopt_data.incremental_outcome.values / nonopt_data.spend.values,
                'mroi_nonoptimized': nonopt_data.mroi.values,
                'pct_of_spend_nonoptimized': nonopt_data.pct_of_spend.values,
                'cpik_nonoptimized': nonopt_data.cpik.values,
            })
            optimized_df_global = pd.concat([optimized_df_global, optimized_df_], ignore_index=True)
    else:
        # global optimization output
        opt_data = dataset_to_dataframe(optimization_results.optimized_data)
        opt_data = opt_data[opt_data['metric'] == 'mean']
        nonopt_data = dataset_to_dataframe(optimization_results.nonoptimized_data)
        nonopt_data = nonopt_data[nonopt_data['metric'] == 'mean']

        optimized_df_global = opt_data.copy()
        optimized_df_global = optimized_df_global.merge(
            nonopt_data.add_suffix('_nonoptimized'), 
            left_on=['channel', 'metric'],
            right_on=['channel_nonoptimized', 'metric_nonoptimized'], 
            how='left'
        )
        optimized_df_global = optimized_df_global.drop(columns=['metric','metric_nonoptimized','channel_nonoptimized'])
        optimized_df_global['geo'] = 'global'

        if 'pct_of_contribution' not in optimized_df_global.columns:
            total_outcome_optimized = np.sum(optimized_df_global['incremental_outcome'])
            optimized_df_global['pct_of_contribution'] = optimized_df_global['incremental_outcome'] / total_outcome_optimized if total_outcome_optimized > 0 else 0
            total_outcome_nonoptimized = np.sum(optimized_df_global['incremental_outcome_nonoptimized'])
            optimized_df_global['pct_of_contribution_nonoptimized'] = optimized_df_global['incremental_outcome_nonoptimized'] / total_outcome_nonoptimized if total_outcome_nonoptimized > 0 else 0

        optimized_df_global['roi_calculated'] = optimized_df_global['incremental_outcome'] / optimized_df_global['spend']
        optimized_df_global['roi_calculated_nonoptimized'] = optimized_df_global['incremental_outcome_nonoptimized'] / optimized_df_global['spend_nonoptimized']

        optimized_df_global = optimized_df_global[['geo', 'channel', 'incremental_outcome', 
                                     'pct_of_contribution', 'spend', 'pct_of_spend', 
                                     'effectiveness', 'roi', 
                                     'roi_calculated', 'mroi', 'cpik', 
                                     'incremental_outcome_nonoptimized', 'pct_of_contribution_nonoptimized', 
                                     'spend_nonoptimized', 
                                     'pct_of_spend_nonoptimized',
                                     'effectiveness_nonoptimized',
                                     'roi_nonoptimized', 
                                     'roi_calculated_nonoptimized', 
                                     'mroi_nonoptimized', 
                                     'cpik_nonoptimized']
                                     ]

    if adjust_negative_values:
        numeric_cols = [col for col in optimized_df_global.columns if col not in ['geo', 'channel']]
        for col in numeric_cols:
            optimized_df_global[col] = optimized_df_global[col].apply(lambda x: 0 if (x < 0 or np.isinf(x)) else x)
    
    return optimized_df_global


def get_optimized_df_old(
    optimization_results: Union[OptimizationResults, Dict[str, OptimizationResults]],
    adjust_negative_values: bool = True
) -> pd.DataFrame:
    """Convert optimization results to a pandas DataFrame.

    This function takes optimization results from either global or geo-specific
    optimization and converts them into a pandas DataFrame for easier analysis.
    It handles both single-geo and multi-geo optimization results.

    Args:
        optimization_results: Either a single OptimizationResults object for global
            optimization, or a dictionary mapping geo names to OptimizationResults
            objects for geo-specific optimization.
        adjust_negative_values: If True, replaces negative and infinite values
            with 0 in the output DataFrame. Defaults to True.

    Returns:
        A pandas DataFrame containing the optimization results with the following
        columns:
        - geo: Geography identifier ('global' for global optimization)
        - channel: Media channel name
        - incremental_outcome: Optimized incremental outcome
        - pct_of_contribution: Percentage contribution to total outcome
        - spend: Optimized spend
        - roi: Return on investment
        - roi_calculated: Calculated ROI (incremental_outcome/spend)
        - mroi: Marginal return on investment
        - cpik: Cost per incremental KPI
        - pct_of_spend: Percentage of total spend
        - effectiveness: Channel effectiveness
        - *_nonoptimized: Corresponding metrics for non-optimized scenario

    """
    optimized_df_global = pd.DataFrame()

    if hasattr(optimization_results, 'keys'):
        # geo optimization output
        for geo in optimization_results.keys():
            opt_data = optimization_results[geo].optimized_data
            nonopt_data = optimization_results[geo].nonoptimized_data

            if hasattr(opt_data, 'pct_of_contribution') and hasattr(nonopt_data, 'pct_of_contribution'):
                pct_of_contribution_optimized = opt_data.pct_of_contribution.values
                pct_of_contribution_nonoptimized = nonopt_data.pct_of_contribution.values
            else:
                total_outcome_optimized = np.sum(opt_data.incremental_outcome.values)
                pct_of_contribution_optimized = opt_data.incremental_outcome.values / total_outcome_optimized if total_outcome_optimized > 0 else np.zeros_like(opt_data.incremental_outcome.values)
                total_outcome_nonoptimized = np.sum(nonopt_data.incremental_outcome.values)
                pct_of_contribution_nonoptimized = nonopt_data.incremental_outcome.values / total_outcome_nonoptimized if total_outcome_nonoptimized > 0 else np.zeros_like(nonopt_data.incremental_outcome.values)

            optimized_df_ = pd.DataFrame({
                'geo': geo,
                'channel': opt_data.channel.values,
                'incremental_outcome': opt_data.incremental_outcome.values,
                'pct_of_contribution': pct_of_contribution_optimized,
                'spend': opt_data.spend.values,
                'roi': opt_data.roi.values,
                'roi_calculated': opt_data.incremental_outcome.values / opt_data.spend.values,
                'mroi': opt_data.mroi.values,
                'cpik': opt_data.cpik.values,
                'pct_of_spend': opt_data.pct_of_spend.values,
                'incremental_outcome_nonoptimized': nonopt_data.incremental_outcome.values,
                'pct_of_contribution_nonoptimized': pct_of_contribution_nonoptimized,
                'spend_nonoptimized': nonopt_data.spend.values,
                'roi_nonoptimized': nonopt_data.roi.values,
                'roi_calculated_nonoptimized': nonopt_data.incremental_outcome.values / nonopt_data.spend.values,
                'mroi_nonoptimized': nonopt_data.mroi.values,
                'pct_of_spend_nonoptimized': nonopt_data.pct_of_spend.values,
                'cpik_nonoptimized': nonopt_data.cpik.values,
            })
            optimized_df_global = pd.concat([optimized_df_global, optimized_df_], ignore_index=True)
    else:
        # global optimization output
        opt_data = optimization_results.optimized_data
        nonopt_data = optimization_results.nonoptimized_data

        if hasattr(opt_data, 'pct_of_contribution') and hasattr(nonopt_data, 'pct_of_contribution'):
            pct_of_contribution_optimized = opt_data.pct_of_contribution.sel(metric='mean').values
            pct_of_contribution_nonoptimized = nonopt_data.pct_of_contribution.sel(metric='mean').values
        else:
            total_outcome_optimized = np.sum(opt_data.incremental_outcome.sel(metric='mean').values)
            pct_of_contribution_optimized = opt_data.incremental_outcome.sel(metric='mean').values / total_outcome_optimized if total_outcome_optimized > 0 else 0
            total_outcome_nonoptimized = np.sum(nonopt_data.incremental_outcome.sel(metric='mean').values)
            pct_of_contribution_nonoptimized = nonopt_data.incremental_outcome.sel(metric='mean').values / total_outcome_nonoptimized if total_outcome_nonoptimized > 0 else 0


        optimized_df_ = pd.DataFrame({
            'geo': 'global',
            'channel': opt_data.channel.values,
            'incremental_outcome': opt_data.incremental_outcome.sel(metric='mean'),
            'pct_of_contribution': pct_of_contribution_optimized,
            'spend': opt_data.spend,
            'pct_of_spend': opt_data.pct_of_spend,
            'effectiveness': opt_data.effectiveness.sel(metric='mean'),
            'roi': opt_data.roi.sel(metric='mean'),
            'roi_calculated': opt_data.incremental_outcome.sel(metric='mean') / opt_data.spend,
            'mroi': opt_data.mroi.sel(metric='mean'), 
            'cpik': opt_data.cpik.sel(metric='mean'),
            'incremental_outcome_nonoptimized': nonopt_data.incremental_outcome.sel(metric='mean'),
            'pct_of_contribution_nonoptimized': pct_of_contribution_nonoptimized,
            'spend_nonoptimized': nonopt_data.spend,
            'pct_of_spend_nonoptimized': nonopt_data.pct_of_spend,
            'effectiveness_nonoptimized': nonopt_data.effectiveness.sel(metric='mean'),
            'roi_nonoptimized': nonopt_data.roi.sel(metric='mean'),
            'roi_calculated_nonoptimized': nonopt_data.incremental_outcome.sel(metric='mean') / nonopt_data.spend,
            'mroi_nonoptimized': nonopt_data.mroi.sel(metric='mean'),
            'cpik_nonoptimized': nonopt_data.cpik.sel(metric='mean'),
        })
        optimized_df_global = pd.concat([optimized_df_global, optimized_df_], ignore_index=True)

    if adjust_negative_values:
        numeric_cols = [col for col in optimized_df_global.columns if col not in ['geo', 'channel']]
        for col in numeric_cols:
            optimized_df_global[col] = optimized_df_global[col].apply(lambda x: 0 if (x < 0 or np.isinf(x)) else x)
    
    return optimized_df_global  

def get_summary_df_old(model_analysis: analyzer.Analyzer, 
                   constants: constants, 
                   adjust_negative_values: bool = True,
                   geos_selected = None,
                   times_selected = None) -> pd.DataFrame:
    """Convert summary metrics output to a pandas DataFrame.

    This function takes a ModelAnalyzer object and converts the summary metrics
    output into a pandas DataFrame for easier analysis and visualization.

    Args:   
        model_analysis: ModelAnalyzer object
        constants: constants object
        adjust_negative_values: If True, replaces negative and infinite values
            with 0 in the output DataFrame. Defaults to True.

    Returns:
        A pandas DataFrame containing the summary metrics.  
    """
    # Summary metrics output with all channels
    summary_metrics_output = model_analysis.summary_metrics(
        marginal_roi_by_reach = False,
        selected_geos = None,
        selected_times = None,
        aggregate_geos = False, 
        aggregate_times = False,
        optimal_frequency = None,
        use_kpi  = False,
        confidence_level = constants.DEFAULT_CONFIDENCE_LEVEL,
        batch_size = constants.DEFAULT_BATCH_SIZE,
        include_non_paid_channels = True,# False
    )

    # Summary metrics output with paid media only
    summary_paid_output = model_analysis.summary_metrics(
        marginal_roi_by_reach = False,
        selected_geos = None,
        selected_times = None,
        aggregate_geos = False, 
        aggregate_times = False,
        optimal_frequency = None,
        use_kpi  = False,
        confidence_level = constants.DEFAULT_CONFIDENCE_LEVEL,
        batch_size = constants.DEFAULT_BATCH_SIZE,
        include_non_paid_channels = False
    )

    # All channels data frame built from summary_metrics_output
    # no paid media metrics are included
    summary_df_base = pd.DataFrame()

    for geo in summary_metrics_output.geo.values:
        for time in summary_metrics_output.time.values:
            for channel in summary_metrics_output.channel.values:
                row = {
                    'time': time,
                    'geo': geo,
                    'channel': channel,
                    'incremental_outcome': float(summary_metrics_output.incremental_outcome.sel(
                        geo=geo, time=time, channel=channel, metric='mean', distribution='posterior').values),
                    'pct_of_contribution': float(summary_metrics_output.pct_of_contribution.sel(
                        geo=geo, time=time, channel=channel, metric='mean', distribution='posterior').values),
                }
                summary_df_base = pd.concat([summary_df_base, pd.DataFrame([row])], ignore_index=True)

    # Paid only data frame built from summary_paid_output
    summary_df_paid_only = pd.DataFrame()

    for geo in summary_paid_output.geo.values:
        for time in summary_paid_output.time.values:
            for channel in summary_paid_output.channel.values:
                row = {
                    'time': time,
                    'geo': geo,
                    'channel': channel,
                    # just get what's relevant for paid media
                    'spend': float(summary_paid_output.spend.sel(
                        geo=geo, time=time, channel=channel).values),
                    'pct_of_spend': float(summary_paid_output.pct_of_spend.sel(
                        geo=geo, time=time, channel=channel).values),                
                    'impressions': float(summary_paid_output.impressions.sel(
                        geo=geo, time=time, channel=channel).values),
                    'pct_of_impressions': float(summary_paid_output.pct_of_impressions.sel(
                        geo=geo, time=time, channel=channel).values),  
                }
                summary_df_paid_only = pd.concat([summary_df_paid_only, pd.DataFrame([row])], ignore_index=True)

    summary_df = pd.merge(summary_df_base, summary_df_paid_only, on=['time', 'geo', 'channel'], how='left')

    # Adjust negative values
    if adjust_negative_values:
        summary_df.loc[summary_df['incremental_outcome'] < 0, 'incremental_outcome'] = 0
        summary_df.loc[summary_df['pct_of_contribution'] < 0, 'pct_of_contribution'] = 0

    summary_df = summary_df[summary_df['channel'] != 'All Channels']

    # # Get baseline, actual and expected values
    expected_vs_actual_output = model_analysis.expected_vs_actual_data(
        aggregate_geos = False,
        aggregate_times = False,
        split_by_holdout_id = False
    )

    if times_selected is None:
        times_selected = expected_vs_actual_output.time.values
    if geos_selected is None:
        geos_selected = expected_vs_actual_output.geo.values

    df_actual_vs_predited_with_analyzer = pd.DataFrame()
    for geo in geos_selected:
        for time in times_selected:
            row = {
                'time': time,
                'geo': geo,
                'expected': float(expected_vs_actual_output.expected.sel(
                    geo=geo, time=time, metric='mean').values),
                'baseline': float(expected_vs_actual_output.baseline.sel(
                    geo=geo, time=time, metric='mean').values),
                'actual': float(expected_vs_actual_output.actual.sel(
                    geo=geo, time=time).values)
            }
            df_actual_vs_predited_with_analyzer = pd.concat([df_actual_vs_predited_with_analyzer, pd.DataFrame([row])], ignore_index=True)
            
    summary_df_subset = summary_df.copy()
    summary_df_subset = summary_df_subset[summary_df_subset['time'].isin(times_selected)]
    full_df = summary_df_subset.merge(df_actual_vs_predited_with_analyzer, on=['time', 'geo'], how='left')
    full_df.head()

    return full_df

def get_summary_df(model_analysis: analyzer.Analyzer, 
                   constants: constants, 
                   adjust_negative_values: bool = True,
                   include_baseline: bool = True,
                   aggregate_geos = False
                   ) -> pd.DataFrame:
    """Convert summary metrics output to a pandas DataFrame.

    This function takes a ModelAnalyzer object and converts the summary metrics
    output into a pandas DataFrame for easier analysis and visualization.

    Args:   
        model_analysis: ModelAnalyzer object
        constants: constants object
        adjust_negative_values: If True, replaces negative and infinite values
            with 0 in the output DataFrame. Defaults to True.
        include_baseline: If True, includes the baseline as a channel and not as
        a separate column in the output DataFrame. Defaults to True.
        aggregate_geos: If True, aggregates the geos into a single national geo.
            Defaults to False.
    Example:
        >>> # For global optimization
        >>> df = get_summary_df(model_analysis, constants, include_baseline=True, aggregate_geos=True)
        >>> # For geo-specific optimization
        >>> df = get_summary_df(model_analysis, constants, include_baseline=True, aggregate_geos=False)

    Returns:
        A pandas DataFrame containing the summary metrics.  
    """
    # Summary metrics output with all channels
    summary_metrics_output = model_analysis.summary_metrics(
        marginal_roi_by_reach = False,
        selected_geos = None,
        selected_times = None,
        aggregate_geos = aggregate_geos, 
        aggregate_times = False,
        optimal_frequency = None,
        use_kpi  = False,
        confidence_level = constants.DEFAULT_CONFIDENCE_LEVEL,
        batch_size = constants.DEFAULT_BATCH_SIZE,
        include_non_paid_channels = True,# False
    )

    # Summary metrics output with paid media only
    summary_paid_output = model_analysis.summary_metrics(
        marginal_roi_by_reach = False,
        selected_geos = None,
        selected_times = None,
        aggregate_geos = aggregate_geos, 
        aggregate_times = False,
        optimal_frequency = None,
        use_kpi  = False,
        confidence_level = constants.DEFAULT_CONFIDENCE_LEVEL,
        batch_size = constants.DEFAULT_BATCH_SIZE,
        include_non_paid_channels = False,
    )

    # All channels data frame built from summary_metrics_output
    # no paid media metrics are included



    summary_df_base = dataset_to_dataframe(summary_metrics_output)
    if aggregate_geos:
        summary_df_base['geo'] = 'national_geo'

    summary_df_base = summary_df_base[
        (summary_df_base['channel'] != 'All Channels') & 
        (summary_df_base['metric'] == 'mean') &
        (summary_df_base['distribution'] == 'posterior')
    ][['time', 'geo','channel', 'incremental_outcome', 'pct_of_contribution']]

    summary_df_paid_only = dataset_to_dataframe(summary_paid_output)
    if aggregate_geos:
        summary_df_paid_only['geo'] = 'national_geo'

    summary_df_paid_only = summary_df_paid_only[
        (summary_df_paid_only['channel'] != 'All Channels') & 
        (summary_df_paid_only['metric'] == 'mean') &
        (summary_df_paid_only['distribution'] == 'posterior')
    ][['time', 'geo', 'channel', 'spend', 'pct_of_spend', 'impressions', 'pct_of_impressions']]

    # merge the two data frames
    summary_df = pd.merge(summary_df_base, summary_df_paid_only, on=['time', 'geo', 'channel'], how='left')


    # Adjust negative values
    if adjust_negative_values:
        summary_df.loc[summary_df['incremental_outcome'] < 0, 'incremental_outcome'] = 0
        summary_df.loc[summary_df['pct_of_contribution'] < 0, 'pct_of_contribution'] = 0

    summary_df = summary_df[summary_df['channel'] != 'All Channels']

    # # Get baseline, actual and expected values
    expected_vs_actual_output = model_analysis.expected_vs_actual_data(
        aggregate_geos = aggregate_geos,
        aggregate_times = False,
        split_by_holdout_id = False
    )

    # convert xarray to dataframe
    df_actual_vs_predited_with_analyzer = dataset_to_dataframe(expected_vs_actual_output)
    if aggregate_geos:
        df_actual_vs_predited_with_analyzer['geo'] = 'national_geo'

    df_actual_vs_predited_with_analyzer = df_actual_vs_predited_with_analyzer[
        (df_actual_vs_predited_with_analyzer['metric'] == 'mean')
    ][['time', 'geo', 'expected', 'baseline', 'actual']]
        
    full_df = summary_df.merge(df_actual_vs_predited_with_analyzer, on=['time', 'geo'], how='left')

    # Adjust negative values
    if adjust_negative_values:
        numeric_cols = [col for col in full_df.columns if col not in ['geo', 'channel', 'time']]
        for col in numeric_cols:
            full_df[col] = full_df[col].apply(lambda x: 0 if (x < 0 or np.isinf(x)) else x)

    if include_baseline:
        # Separate df for baseline
        baseline_df = full_df.copy()
        baseline_df = baseline_df.groupby(['time', 'geo']).agg({
            'baseline': 'mean', 
            'actual': 'mean', 
            'expected': 'mean',
            'pct_of_contribution': 'sum'
        }).reset_index()
        baseline_df.rename(columns={'baseline': 'incremental_outcome'}, inplace=True)
        baseline_df['channel'] = 'Baseline'

        # Calculate pct_of_contribution for the baseline
        baseline_df['pct_of_contribution'] = 100 - baseline_df['pct_of_contribution']

        # # Append baseline_df to full_df
        # non-matching columns will be filled with NaN values
        full_df_modified = full_df.copy()
        full_df_modified = pd.concat([full_df_modified, baseline_df], ignore_index=True)

        # # drop the baseline column
        full_df_modified = full_df_modified.drop(columns=['baseline'])

        full_df_modified = full_df_modified.sort_values(by=['time', 'geo', 'channel']).reset_index(drop=True)
    else:
        full_df_modified = full_df

    return full_df_modified

def get_response_curves_df(model_analysis, 
                          geos_selected: list,
                          spend_multipliers: list = list(np.round(np.arange(0, 2.2, 0.2), 1)),
                          time_slices_dict: dict = None,
                          multiprocess: bool = False) -> pd.DataFrame:
    """Get the response curves data frame.

    Args:
        model_analysis: ModelAnalyzer object
        geos_selected: list of geos to include in the response curves
        spend_multipliers: list of spend multipliers to include in the response curves.
            Default is a list of spend multipliers from 0 to 2.2 in steps of 0.2.
        time_slices_dict: dictionary of time slices to include in the response curves
        multiprocess: boolean to use multiprocessing to loop through geos and time slices
    Returns:
        A pandas DataFrame containing the response curves.
    """
 
    response_curves_by_geo = pd.DataFrame()

    if time_slices_dict is None:
        for geo in geos_selected:
            response_curves_data = model_analysis.response_curves(
                selected_geos=[geo],
                spend_multipliers=spend_multipliers,
                )
            response_curves_data_df = dataset_to_dataframe(response_curves_data)
            response_curves_data_df['geo'] = geo
            response_curves_by_geo = pd.concat([response_curves_by_geo, response_curves_data_df], ignore_index=True)
    elif multiprocess == True:
        response_curves_by_geo = pd.DataFrame()
        from concurrent.futures import ThreadPoolExecutor
        import concurrent.futures

        def process_geo_time_slice(args):
            geo, time_slice = args
            response_curves_data = model_analysis.response_curves(
                selected_geos=[geo],
                spend_multipliers=spend_multipliers,
                selected_times=time_slices_dict[time_slice],
            )
            response_curves_data_df = dataset_to_dataframe(response_curves_data)
            response_curves_data_df['geo'] = geo
            response_curves_data_df['time_slice'] = time_slice
            return response_curves_data_df

        # Create list of argument tuples for parallel processing
        combinations = [(geo, time_slice) 
                       for time_slice in time_slices_dict.keys() 
                       for geo in geos_selected]

        results = []
        # Process combinations in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(process_geo_time_slice, combo) for combo in combinations]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_result = future.result()
                    results.append(batch_result)
                except Exception as e:
                    print(f"Error processing a batch: {e}")

        # Combine all results
        response_curves_by_geo = pd.concat(results, ignore_index=True)
    else:
        response_curves_by_geo = pd.DataFrame()
        for time_slice in time_slices_dict.keys():
            for geo in geos_selected:
                response_curves_data = model_analysis.response_curves(
                    selected_geos=[geo],
                    spend_multipliers=spend_multipliers,
                    selected_times=time_slices_dict[time_slice],
                )
                response_curves_data_df = dataset_to_dataframe(response_curves_data)
                response_curves_data_df['geo'] = geo
                response_curves_data_df['time_slice'] = time_slice
                response_curves_by_geo = pd.concat([response_curves_by_geo, response_curves_data_df], ignore_index=True)

    # spread metric to columns so that incremental_outcome will have three columns: incremental_outcome_mean, incremental_outcome_ci_lo and incremental_outcome_ci_hi
    if time_slices_dict is None:
        response_curves_by_geo = response_curves_by_geo.pivot_table(
            index=['geo', 'channel', 'spend_multiplier','spend'], 
            columns='metric',
            values='incremental_outcome').reset_index()
    else:
        response_curves_by_geo = response_curves_by_geo.pivot_table(
            index=['geo', 'channel', 'spend_multiplier','spend', 'time_slice'], 
            columns='metric', 
            values='incremental_outcome').reset_index()
    response_curves_by_geo = response_curves_by_geo.rename(columns={
        'mean': 'incremental_outcome_mean',
        'ci_lo': 'incremental_outcome_ci_lo',
        'ci_hi': 'incremental_outcome_ci_hi'
    })
    response_curves_by_geo.columns.name = None
    return response_curves_by_geo

def export_results(full_df, 
                   response_curves_by_geo, 
                   optimized_df_geo, 
                   fname_components = [None, None], 
                   to_excel = True
                   ):
    """Export optimization results to Excel or CSV files.

    This function exports the optimization results to Excel or CSV files.
    It allows for exporting to Excel or CSV files, and for specifying the
    filename components (date and client name).

    Args:
        full_df: pandas DataFrame containing the optimization results
        response_curves_by_geo: pandas DataFrame containing the response curves
        optimized_df_geo: pandas DataFrame containing the optimization results
        fname_components: list of two elements, the first is the date and the second is the client name
        to_excel: boolean, if True, the results are exported to Excel, otherwise they are exported to CSV
    """
    # date
    if fname_components[0] is None:
        today = datetime.today()
        fname_components[0] = today.strftime('%Y%m%d')

    # client name
    if fname_components[1] is None:
       fname_components[1] = 'demo'  

    if to_excel:
    # export to excel
        try:
            with pd.ExcelWriter(f'{fname_components[1]}_{fname_components[0]}_output.xlsx') as writer:
                if full_df is not None:
                    full_df.to_excel(writer, sheet_name='mmm', index=False)
                if response_curves_by_geo is not None:
                    response_curves_by_geo.to_excel(writer, sheet_name='response_curves', index=False)
                if optimized_df_geo is not None:
                    optimized_df_geo.to_excel(writer, sheet_name='optimization', index=False)
            print(f"Data exported successfully to {fname_components[1]}_{fname_components[0]}_output.xlsx")
        except Exception as e:
            print(f"Error exporting to Excel: {e}")
    else:
        # Export to CSV files
        if full_df is not None:
            full_df.to_csv(f'mmm_output_{fname_components[1]}_{fname_components[0]}.csv', index=False)
            print(f"Full data exported to mmm_output_{fname_components[1]}_{fname_components[0]}.csv")
            
        if response_curves_by_geo is not None:
            response_curves_by_geo.to_csv(f'response_curves_{fname_components[1]}_{fname_components[0]}.csv', index=False)
            print(f"Response curves exported to response_curves_{fname_components[1]}_{fname_components[0]}.csv")
            
        if optimized_df_geo is not None:
            optimized_df_geo.to_csv(f'optimization_{fname_components[1]}_{fname_components[0]}.csv', index=False)
            print(f"Optimization data exported to optimization_{fname_components[1]}_{fname_components[0]}.csv")


def upload_csv_to_gcs(bucket_name, file_name, df):
    """Uploads a data frame to Google Cloud Storage.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        file_name (str): Name of the file to create in GCS (without .csv extension)
        df (pd.DataFrame): DataFrame to upload
        
    Returns:
        None
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Create blob with full path including .csv extension
    blob = bucket.blob(f"{file_name}.csv")
    
    # Convert DataFrame to CSV string and upload directly
    csv_data = df.to_csv(index=False)
    blob.upload_from_string(csv_data)
    
    print(f"File '{file_name}.csv' uploaded to '{bucket_name}'.")

def create_bq_schema(df: pd.DataFrame):
    """Create a BigQuery schema from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to create the schema from
        
    Returns:
        list: A list of BigQuery SchemaField objects
    """
    from google.cloud import bigquery

    pandas_bq_dtype_map = {
        'int64': 'INTEGER',
        'float64': 'FLOAT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'DATETIME',
        'object': 'STRING'
    }
    # Any pandas dtype containing 'datetime' maps to 'DATETIME'
    # otherwise use the pandas_bq_dtype_map
    def map_pandas_dtype_to_bq(dtype_str):
        if 'datetime' in dtype_str:
            return 'DATETIME'
        return pandas_bq_dtype_map.get(dtype_str, 'STRING') # default to STRING if not in map
    schema = []
    for col, dtype in df.dtypes.items():
        bq_type = map_pandas_dtype_to_bq(str(dtype))
        schema.append(bigquery.SchemaField(col, bq_type))
    return schema


def upload_to_bq(
    data,
    project_id: str = "ag-stage-app-prod", 
    dataset_id: str = "amanresorts_amanresorts_rrpn", 
    table_id: str = "mmm_data_platform_gqv",
    date_field: str | None = None,  # Must be a DATE or DATETIME column for partitioning
    schema=None,
    write_disposition: str = "WRITE_TRUNCATE",  # "WRITE_TRUNCATE": replace all data; "WRITE_APPEND": append data
    clustering_fields: list | None = None,
):
    """
    Uploads a pandas DataFrame to a Google BigQuery table, with optional table creation,
    date-based partitioning, clustering, and schema specification.

    Args:
        data (pd.DataFrame): 
            The DataFrame to upload to BigQuery.
        project_id (str, optional): 
            GCP project ID containing the destination dataset and table. 
            Defaults to "ag-stage-app-prod".
        dataset_id (str, optional): 
            The BigQuery dataset ID where the table resides or will be created. 
            Defaults to "amanresorts_amanresorts_rrpn".
        table_id (str, optional): 
            The BigQuery table ID (name) to write to. 
            Defaults to "mmm_data_platform_gqv".
        date_field (str, optional): 
            (Recommended) Name of a DATE or DATETIME column in the DataFrame to use for 
            partitioning. If specified, the BigQuery table will be partitioned by this column.
            Defaults to None (no partitioning).
        schema (List[google.cloud.bigquery.SchemaField], optional): 
            Schema for the destination table. If None, schema is inferred from the DataFrame.
        write_disposition (str, optional): 
            BigQuery write mode; "WRITE_TRUNCATE" (replace table contents), "WRITE_APPEND" 
            (add to existing data), or "WRITE_EMPTY" (fail if not empty). 
            Defaults to "WRITE_TRUNCATE".
        clustering_fields (list, optional): 
            List of columns to be used for clustering in BigQuery. Defaults to None.

    Behavior:
        - Checks if the destination table exists. If not, creates it with the specified
          schema, partitioning, and clustering (as specified by arguments).
        - If the table exists, respects the 'write_disposition': overwrites or appends.
        - Uploads data from the DataFrame to the table.
        - Prints status messages along the way.

    Example:
        >>> from google.cloud.bigquery import SchemaField
        >>> schema = [SchemaField("date", "DATE"), SchemaField("value", "FLOAT")]
        >>> upload_to_bq(
        ...     df, 
        ...     project_id="my-proj", 
        ...     dataset_id="my_ds", 
        ...     table_id="my_table",
        ...     date_field="date",
        ...     schema=schema,
        ...     write_disposition="WRITE_TRUNCATE"
        ... )

    Raises:
        Any BigQuery client or job errors encountered during table creation or upload.

    """

    from google.cloud import bigquery
    client = bigquery.Client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"

    if schema is None:
        schema = create_bq_schema(data)

    # Define partitioning and clustering
    table = bigquery.Table(table_ref, schema=schema)
    if date_field:
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=date_field,
        )
    if clustering_fields:
        table.clustering_fields = clustering_fields

    # Check if table exists, else create
    try:
        client.get_table(table_ref)
        if write_disposition == "WRITE_TRUNCATE":
            data_action = "overwriting"
        elif write_disposition == "WRITE_APPEND":
            data_action = "appending"
        else:
            data_action = "writing"
        print(f"Table {table_ref} exists, {data_action} data...")
    except Exception as e:
        print(f"Table {table_ref} does not exist. Creating it...")
        table = client.create_table(table)
        print(f"Created table: {table.full_table_id}")

    # Upload DataFrame
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        write_disposition=write_disposition
    )
    load_job = client.load_table_from_dataframe(data, table_ref, job_config=job_config)
    load_job.result()  # Wait for the job to complete

    print(f"Wrote {load_job.output_rows} rows to {table_ref}")

def create_model_diagnostics_table(mmm):
    """Create a model evaluation table by geo.
    
    Args:
        mmm (meridian.Meridian): The Meridian model object
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the model evaluation table
    """
    from meridian.analysis import visualizer
    model_diagnostics = visualizer.ModelDiagnostics(mmm)

    model_diagnostics_df = model_diagnostics.predictive_accuracy_table(
        selected_geos = None,
        selected_times = None,
        column_var = None,
    )
    model_diagnostics_df['geo'] = 'global'
    # Create a model evaluation table by geo
    model_diagnostics_geo_df = pd.DataFrame()
    for region in mmm.input_data.geo.values.tolist():
        print(f"Geo {region} processing")
        md_geo_df = model_diagnostics.predictive_accuracy_table(
            selected_geos = [region],
            selected_times = None,
            column_var = None,
        )
        md_geo_df['geo'] = region
        # remove national rows
        md_geo_df = md_geo_df[md_geo_df['geo_granularity'] == 'geo']
        model_diagnostics_geo_df = pd.concat([model_diagnostics_geo_df, md_geo_df], ignore_index=True)
        
    # append geo results to model_diagnostics_df
    model_diagnostics_df = pd.concat([model_diagnostics_df, model_diagnostics_geo_df], ignore_index=True)
    return model_diagnostics_df

def save_mmm(mmm, file_path: str):
  """Save the model object to a `pickle` file path.

  WARNING: There is no guarantee for future compatibility of the binary file
  output of this function. We recommend using `load_mmm()` with the same
  version of the library that was used to save the model.

  Args:
    mmm: Model object to save.
    file_path: File path to save a pickled model object.
  """
  import os
  if not os.path.exists(os.path.dirname(file_path)):
    os.makedirs(os.path.dirname(file_path))

  with open(file_path, "wb") as f:
    joblib.dump(mmm, f)
