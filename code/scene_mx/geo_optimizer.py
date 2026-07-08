"""Module containing geo-specific optimization functionality."""

import warnings
import os

# Suppress warnings before importing libraries
warnings.filterwarnings('ignore', category=UserWarning)  # Suppress UserWarnings (e.g., from meridian about media_spend)
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress RuntimeWarnings (e.g., divide by zero)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

from meridian.analysis import analyzer
from meridian.analysis import optimizer
import multiprocessing as mp
from meridian.analysis.optimizer import (
    BudgetOptimizer,
    OptimizationResults,
    get_round_factor,
    get_optimization_bounds,
    _SpendConstraint,
    _exceeds_optimization_constraints,
    _validate_pct_of_spend
)
from meridian import constants as c
import tensorflow as tf
import numpy as np
import xarray as xr
from typing import Any, Mapping, Sequence
from dataclasses import dataclass

# Suppress TensorFlow warnings after import
if hasattr(tf, 'get_logger'):
    tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

# Configure TensorFlow for better performance
# Use multiple threads for intra-op parallelism (operations within a single op)
# Use multiple threads for inter-op parallelism (parallel execution of independent ops)
def configure_tensorflow_threads():
    """Configure TensorFlow to use multiple threads efficiently."""
    try:
        # Get number of CPU cores
        import os
        num_threads = int(os.environ.get('TF_NUM_THREADS', os.cpu_count() or 4))
        
        # Configure TensorFlow threading
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        
        # Enable optimizations (if available)
        try:
            tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation if available
        except AttributeError:
            pass  # XLA not available
    except Exception:
        pass  # Ignore if configuration fails

# Configure TensorFlow on module load
configure_tensorflow_threads()


@dataclass
class FixedBudgetScenario:
    """A fixed budget optimization scenario.

    Attributes:
        total_budget: The total budget for the optimization period. Must be
            non-negative. If unspecified, it represents historical total spend.
    """

    total_budget: float | None = None

    def __post_init__(self):
        if self.total_budget is not None and self.total_budget < 0:
            raise ValueError("Total budget must be non-negative.")


@dataclass
class FlexibleBudgetScenario:
    """A flexible budget optimization scenario.

    Attributes:
        target_metric: The target metric to optimize for. This should be ROI or
            mROI.
        target_value: The target value for the above metric. Must be non-negative.
    """

    target_metric: str
    target_value: float

    def __post_init__(self):
        if self.target_value < 0:
            raise ValueError("Target value must be non-negative.")
        if self.target_metric not in [c.ROI, c.MROI]:
            raise ValueError(
                f"Target metric must be one of {[c.ROI, c.MROI]}, "
                f"got {self.target_metric}."
            )


@dataclass
class GeoOptimizationGrid:
    """Optimization grid information with geo-specific functionality.

    Attributes:
        historical_spend: ndarray of shape `(n_paid_channels,)` containing
            aggregated historical spend allocation for spend for all media and RF
            channels. The order matches `InputData.get_all_paid_channels`.
        use_kpi: Whether using generic KPI or revenue.
        use_posterior: Whether posterior distributions were used, or prior.
        use_optimal_frequency: Whether optimal frequency was used.
        round_factor: The round factor used for the optimization grid.
        optimal_frequency: Optional ndarray of shape `(n_paid_channels,)`,
            containing the optimal frequency per channel. Value is `None` if the model
            does not contain reach and frequency data, or if the model does contain
            reach and frequency data, but historical frequency is used for the
            optimization scenario.
        selected_times: The time coordinates from the model used in this grid.
        selected_geo: The selected geography for optimization.
    """

    _grid_dataset: xr.Dataset

    historical_spend: np.ndarray
    use_kpi: bool
    use_posterior: bool
    use_optimal_frequency: bool
    round_factor: int
    optimal_frequency: np.ndarray | None
    selected_times: list[str] | None
    selected_geo: str | None = None
    step_size: float = 1.0

    @property
    def grid_dataset(self) -> xr.Dataset:
        """Gets the grid dataset."""
        return self._grid_dataset

    @property
    def spend_grid(self) -> np.ndarray:
        """Gets the spend grid."""
        return self._grid_dataset.spend.values

    @property
    def incremental_outcome_grid(self) -> np.ndarray:
        """Gets the incremental outcome grid."""
        return self._grid_dataset.incremental_outcome.values

    @property
    def spend_step_size(self) -> float:
        """Gets the spend step size."""
        # Ensure we handle negative round factors correctly for precision
        return self.step_size

    def optimize(
        self,
        scenario: FixedBudgetScenario | FlexibleBudgetScenario,
        n_samples: int = 1000,
        verbose: bool = False,
    ) -> np.ndarray:
        """Optimizes spend allocation based on the scenario.

        Args:
            scenario: The optimization scenario.
            n_samples: Number of random samples to generate for the spend grid.
            verbose: Whether to print verbose output (default: False).

        Returns:
            The optimal spend allocation.
        """
        spend_grid = self.spend_grid
        incremental_outcome_grid = self.incremental_outcome_grid
        n_channels = spend_grid.shape[1]

        # For each point in the grid, calculate total spend and outcome
        total_spend = np.sum(spend_grid, axis=1)
        total_outcome = incremental_outcome_grid

        if isinstance(scenario, FixedBudgetScenario):
            # For fixed budget, find the point that maximizes outcome while meeting budget
            target_budget = scenario.total_budget or np.sum(self.historical_spend)
            
            # Use 1% of target budget as tolerance to account for rounding
            tolerance = max(self.spend_step_size, target_budget * 0.01)
            if verbose:
                print(f"Target budget: {target_budget}")
                print(f"Total spend range: {np.min(total_spend)} to {np.max(total_spend)}")
                print(f"Using tolerance: {tolerance}")
            
            valid_points = np.abs(total_spend - target_budget) <= tolerance
            if not np.any(valid_points):
                # Find the closest point to the target budget
                closest_idx = np.argmin(np.abs(total_spend - target_budget))
                closest_budget = total_spend[closest_idx]
                diff = abs(closest_budget - target_budget)
                pct_diff = diff / target_budget * 100 if target_budget else 0
                if verbose:
                    print(f"WARNING: No grid point within tolerance. Using closest available budget: {closest_budget} (diff: {diff}, {pct_diff:.2f}% of target)")
                return spend_grid[closest_idx]
            else:
            # Among valid points, choose one with highest outcome
                valid_outcomes = total_outcome[valid_points]
                max_outcome_idx = np.argmax(valid_outcomes)
                best_point = np.where(valid_points)[0][max_outcome_idx]
                return spend_grid[best_point]

        else:
            # For flexible budget, find point that meets ROI/mROI target with highest spend
            target_metric = scenario.target_metric
            target_value = scenario.target_value

            # Calculate ROI/mROI for each point
            roi = np.zeros_like(total_outcome)
            nonzero_spend = total_spend > 0
            roi[nonzero_spend] = total_outcome[nonzero_spend] / total_spend[nonzero_spend]
            
            if target_metric == c.MROI:
                # For mROI, we need to calculate the marginal ROI
                # Sort points by total spend
                sort_idx = np.argsort(total_spend)
                sorted_spend = total_spend[sort_idx]
                sorted_outcome = total_outcome[sort_idx]
                
                # Calculate marginal changes
                spend_diff = np.diff(sorted_spend)
                outcome_diff = np.diff(sorted_outcome)
                
                # Calculate mROI where spend difference is nonzero
                mroi = np.zeros_like(total_outcome)
                nonzero_diff = spend_diff > 0
                if np.any(nonzero_diff):
                    # Create a mask for valid points where mROI meets target
                    valid_points = np.zeros_like(total_outcome, dtype=bool)
                    valid_indices = np.where(nonzero_diff)[0]
                    mroi_values = outcome_diff[valid_indices] / spend_diff[valid_indices]
                    valid_points[sort_idx[1:][valid_indices]] = mroi_values >= target_value
                else:
                    valid_points = np.zeros_like(total_outcome, dtype=bool)
            else:
                # For ROI, use calculated values directly
                valid_points = roi >= target_value

            if not np.any(valid_points):
                raise ValueError(
                    f"No valid points found meeting target {target_metric} of {target_value}. "
                    f"Maximum achievable {target_metric} is {np.max(roi)}."
                )

            # Among valid points, choose one with highest spend
            valid_spend = total_spend[valid_points]
            max_spend_idx = np.argmax(valid_spend)
            best_point = np.where(valid_points)[0][max_spend_idx]
            return spend_grid[best_point]


def _validate_budget(
    fixed_budget: bool,
    budget: float | None,
    target_roi: float | None,
    target_mroi: float | None,
):
    """Validates budget optimization parameters.

    Args:
        fixed_budget: Whether to use fixed budget optimization.
        budget: The total budget for fixed budget optimization.
        target_roi: The target ROI for flexible budget optimization.
        target_mroi: The target mROI for flexible budget optimization.

    Raises:
        ValueError: If the budget parameters are invalid.
    """
    if fixed_budget and budget is not None and budget < 0:
        raise ValueError("Budget must be non-negative.")
    if not fixed_budget and target_roi is None and target_mroi is None:
        raise ValueError(
            "Must specify either target_roi or target_mroi for flexible budget "
            "optimization."
        )
    if not fixed_budget and target_roi is not None and target_mroi is not None:
        raise ValueError(
            "Cannot specify both target_roi and target_mroi for flexible budget "
            "optimization."
        )
    if not fixed_budget and target_roi is not None and target_roi < 0:
        raise ValueError("Target ROI must be non-negative.")
    if not fixed_budget and target_mroi is not None and target_mroi < 0:
        raise ValueError("Target mROI must be non-negative.")


class GeoAnalyzer(analyzer.Analyzer):
    """Analyzer with geographic filtering capabilities."""
    
    def get_historical_spend(
        self,
        selected_times: Sequence[str] | None,
        selected_geo: str | None = None,
        include_media: bool = True,
        include_rf: bool = True,
    ) -> xr.DataArray:
        """Gets historical spend with geo filtering."""
        # Get spend data directly from Meridian model to include geo dimension
        spend_data = self._meridian.input_data.media_spend
        if selected_times:
            spend_data = spend_data.sel(time=selected_times)
        
        # Filter channels based on include flags
        channel_names = []
        if include_media:
            channel_names.extend(self._meridian.input_data.media.media_channel.values)
        if include_rf:
            channel_names.extend(self._meridian.input_data.rf_channel.values)
            
        spend = spend_data.sel(media_channel=channel_names)
        if selected_geo:
            spend = spend.sel(geo=selected_geo)
        return spend


class BudgetOptimizerGeo(BudgetOptimizer):
    """Budget optimizer with geographic filtering capabilities."""

    def __init__(self, meridian):
        """Initialize BudgetOptimizerGeo.

        Args:
            meridian: A fitted Meridian model.
        """
        super().__init__(meridian)
        self._analyzer = GeoAnalyzer(self._meridian)
        self.selected_geo = None

    def optimize(
        self,
        use_posterior: bool = True,
        selected_times: tuple[str | None, str | None] | None = None,
        selected_geo: str | None = None,
        fixed_budget: bool = True,
        budget: float | None = None,
        pct_of_spend: Sequence[float] | None = None,
        spend_constraint_lower: _SpendConstraint | None = None,
        spend_constraint_upper: _SpendConstraint | None = None,
        target_roi: float | None = None,
        target_mroi: float | None = None,
        gtol: float = 0.0001,
        use_optimal_frequency: bool = True,
        use_kpi: bool = False,
        confidence_level: float = c.DEFAULT_CONFIDENCE_LEVEL,
        batch_size: int = c.DEFAULT_BATCH_SIZE,
        step_size: int = 1000,
        n_samples: int = 1000,
        verbose: bool = False,
    ) -> OptimizationResults:
        """Finds the optimal budget allocation that maximizes outcome for a specific geo."""
        # Store selected_geo as instance variable
        self.selected_geo = selected_geo
        _validate_budget(
            fixed_budget=fixed_budget,
            budget=budget,
            target_roi=target_roi,
            target_mroi=target_mroi,
        )

        if selected_times is not None:
            start_date, end_date = selected_times
            selected_time_dims = self._meridian.expand_selected_time_dims(
                start_date=start_date,
                end_date=end_date,
            )
        else:
            selected_time_dims = None

        hist_spend = self._analyzer.get_historical_spend(
            selected_time_dims,
            selected_geo=selected_geo,
            include_media=self._meridian.n_media_channels > 0,
            include_rf=self._meridian.n_rf_channels > 0,
        ).data

        use_historical_budget = budget is None or round(budget) == round(
            np.sum(hist_spend)
        )
        budget = budget or np.sum(hist_spend)
        pct_of_spend = _validate_pct_of_spend(n_channels=hist_spend.shape[1], hist_spend=hist_spend, pct_of_spend=pct_of_spend)
        spend = budget * pct_of_spend
        round_factor = get_round_factor(budget, gtol)
        rounded_spend = np.round(spend, round_factor).astype(int)

        if self._meridian.n_rf_channels > 0 and use_optimal_frequency:
            optimal_frequency = tf.convert_to_tensor(
                self._analyzer.optimal_freq(
                    use_posterior=use_posterior,
                    selected_times=selected_time_dims,
                    selected_geo=selected_geo,
                    use_kpi=use_kpi,
                ).optimal_frequency,
                dtype=tf.float32,
            )
        else:
            optimal_frequency = None

        # (optimization_lower_bound, optimization_upper_bound, spend_bounds) = (
        (optimization_lower_bound, optimization_upper_bound) = (
            get_optimization_bounds(
                n_channels=hist_spend.shape[1],
                spend=rounded_spend,
                round_factor=round_factor,
                spend_constraint_lower=spend_constraint_lower,
                spend_constraint_upper=spend_constraint_upper,
                # fixed_budget=fixed_budget,
            )
        )

        # print(f"Optimization lower bound shape: {optimization_lower_bound.shape}")
        # print(f"Optimization upper bound shape: {optimization_upper_bound.shape}")
        # print(f"Number of channels: {hist_spend.shape[1]}")
        # print(f"Number of time steps: {hist_spend.shape[0]}")


        optimization_grid = self.create_optimization_grid(
            historical_spend=hist_spend,
            spend_bound_lower=optimization_lower_bound,
            spend_bound_upper=optimization_upper_bound,
            selected_times=selected_time_dims,
            round_factor=round_factor,
            use_posterior=use_posterior,
            use_kpi=use_kpi,
            use_optimal_frequency=use_optimal_frequency,
            optimal_frequency=optimal_frequency,
            batch_size=batch_size,
            step_size=step_size,
            n_samples=n_samples,
        )

        if fixed_budget:
            scenario = FixedBudgetScenario(
                total_budget=budget,
            )
        else:
            scenario = FlexibleBudgetScenario(
                target_metric="roi" if target_roi is not None else "mroi",
                target_value=target_roi or target_mroi,
            )

        optimal_spend = optimization_grid.optimize(scenario)

        # Calculate spend ratio and bounds (handle divide by zero)
        # optimal_spend is 1D (channels), hist_spend is 2D (time, channels)
        # Use safe division that handles broadcasting and zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            spend_ratio = np.divide(optimal_spend, hist_spend)
            spend_ratio = np.where(np.isfinite(spend_ratio), spend_ratio, 0)
        spend_bounds = (optimization_lower_bound, optimization_upper_bound)

        # Create the optimization results object
        results = OptimizationResults(
            meridian=self._meridian,
            analyzer=self._analyzer,
            spend_ratio=spend_ratio,
            spend_bounds=spend_bounds,
            # use_posterior=use_posterior,
            # use_optimal_frequency=use_optimal_frequency,
            _nonoptimized_data=self._create_optimization_results(
                spend=hist_spend,
                selected_times=selected_time_dims,
                selected_geo=selected_geo,
                use_posterior=use_posterior,
                use_kpi=use_kpi,
                use_optimal_frequency=use_optimal_frequency,
                optimal_frequency=optimal_frequency,
                batch_size=batch_size,
            ),
            _nonoptimized_data_with_optimal_freq=self._create_optimization_results(
                spend=hist_spend,
                selected_times=selected_time_dims,
                selected_geo=selected_geo,
                use_posterior=use_posterior,
                use_kpi=use_kpi,
                use_optimal_frequency=True,
                optimal_frequency=optimal_frequency,
                batch_size=batch_size,
            ),
            _optimized_data=self._create_optimization_results(
                spend=optimal_spend,
                selected_times=selected_time_dims,
                selected_geo=selected_geo,
                use_posterior=use_posterior,
                use_kpi=use_kpi,
                use_optimal_frequency=use_optimal_frequency,
                optimal_frequency=optimal_frequency,
                batch_size=batch_size,
            ),
            _optimization_grid=optimization_grid,
        )

        return results

    def create_optimization_grid(
        self,
        historical_spend: np.ndarray,
        spend_bound_lower: np.ndarray,
        spend_bound_upper: np.ndarray,
        selected_times: Sequence[str],
        round_factor: int,
        use_posterior: bool = True,
        use_kpi: bool = False,
        use_optimal_frequency: bool = False,
        optimal_frequency: xr.DataArray | None = None,
        batch_size: int = c.DEFAULT_BATCH_SIZE,
        step_size: int = 1000,
        n_samples: int = 1000,
        verbose: bool = False,
    ) -> GeoOptimizationGrid:
        """Creates an optimization grid for a specific geo.

        Args:
            historical_spend: Historical spend allocation.
            spend_bound_lower: Lower bound for spend.
            spend_bound_upper: Upper bound for spend.
            selected_times: Selected time points.
            round_factor: Round factor for spend values.
            use_posterior: Whether to use posterior distributions.
            use_kpi: Whether to use KPI instead of revenue.
            use_optimal_frequency: Whether to use optimal frequency.
            optimal_frequency: Optional optimal frequency values.
            batch_size: Batch size for computations.
            step_size: Step size for spend values.
            n_samples: Number of random samples to generate for the spend grid.
            verbose: Whether to print verbose output (default: False).

        Returns:
            A GeoOptimizationGrid object.
        """
        # Use fixed step size of 1000 for optimization grid
        # step_size = 1000 # TODO: this may be too high, if more grid points are needed, it should be lowered
        # print(f"Spend step size: {step_size}")
        
        # Sum multi-dimensional bounds across all dimensions except channels
        if spend_bound_upper.ndim > 1:
            # Sum across all dimensions except the last one (channels)
            spend_bound_upper = np.sum(spend_bound_upper, axis=tuple(range(spend_bound_upper.ndim - 1)))
            
        if spend_bound_lower.ndim > 1:
            # Sum across all dimensions except the last one (channels)
            spend_bound_lower = np.sum(spend_bound_lower, axis=tuple(range(spend_bound_lower.ndim - 1)))
            
        # Ensure the bounds match the number of channels in historical_spend
        if historical_spend.ndim == 1:
            n_channels = len(historical_spend)
        else:
            n_channels = historical_spend.shape[-1]

        if verbose:
            print(f"In create_optimization_grid:")
            print(f"spend_bound_lower shape: {spend_bound_lower.shape}")
            print(f"spend_bound_upper shape: {spend_bound_upper.shape}")
            print(f"historical_spend shape: {historical_spend.shape}")
            print(f"n_channels: {n_channels}")
            
        if len(spend_bound_lower) != n_channels or len(spend_bound_upper) != n_channels:
            raise ValueError(
                f"Spend bounds must have same length as historical_spend ({n_channels}), "
                f"got {len(spend_bound_lower)} and {len(spend_bound_upper)}"
            )
        
        # Ensure historical_spend is 1D for grid creation
        if historical_spend.ndim > 1:
            historical_spend = np.sum(historical_spend, axis=tuple(range(historical_spend.ndim - 1)))
            
        # Create the grids with pre-allocated space
        spend_grid, incremental_outcome_grid = self._create_grids(
            spend=historical_spend,
            spend_bound_lower=spend_bound_lower,
            spend_bound_upper=spend_bound_upper,
            step_size=step_size,
            selected_times=selected_times,
            use_posterior=use_posterior,
            use_kpi=use_kpi,
            optimal_frequency=optimal_frequency,
            batch_size=batch_size,
            n_samples=n_samples,
        )

        grid_dataset = self._create_grid_dataset(
            spend_grid=spend_grid,
            spend_step_size=step_size,
            incremental_outcome_grid=incremental_outcome_grid,
        )

        return GeoOptimizationGrid(
            _grid_dataset=grid_dataset,
            historical_spend=historical_spend,
            use_kpi=use_kpi,
            use_posterior=use_posterior,
            use_optimal_frequency=use_optimal_frequency,
            round_factor=round_factor,
            optimal_frequency=optimal_frequency,
            selected_times=selected_times,
            step_size=step_size,
        )

    def _create_grids(
        self,
        spend: np.ndarray,
        spend_bound_lower: np.ndarray,
        spend_bound_upper: np.ndarray,
        step_size: float,
        selected_times: Sequence[str],
        use_posterior: bool = True,
        use_kpi: bool = False,
        optimal_frequency: xr.DataArray | None = None,
        batch_size: int = c.DEFAULT_BATCH_SIZE,
        n_samples: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Creates spend and incremental outcome grids using random sampling.

        Args:
            spend: Historical spend allocation.
            spend_bound_lower: Lower bound for spend.
            spend_bound_upper: Upper bound for spend.
            step_size: Step size for spend values.
            selected_times: Selected time points.
            use_posterior: Whether to use posterior distributions.
            use_kpi: Whether to use KPI instead of revenue.
            optimal_frequency: Optional optimal frequency values.
            batch_size: Batch size for computations.
            n_samples: Number of random samples to generate.

        Returns:
            A tuple of (spend_grid, incremental_outcome_grid).
        """
        n_channels = len(spend)
        rng = np.random.default_rng()
        spend_grid = np.zeros((n_samples, n_channels))
        original_total_spend = np.sum(spend)
        tolerance = original_total_spend * 0.05  # 5% tolerance
        
        max_attempts = n_samples * 30  # Allow up to 30x the number of samples as attempts
        attempts = 0
        valid_spend_samples = []
        while len(valid_spend_samples) < n_samples and attempts < max_attempts:
            sample = []
            for j in range(n_channels):
                possible_values = np.arange(
                    spend_bound_lower[j],
                    spend_bound_upper[j] + step_size,
                    step_size,
                )
                sample.append(rng.choice(possible_values, size=1, replace=True)[0])
            total_spend = np.sum(sample)
            if abs(total_spend - original_total_spend) <= tolerance:
                valid_spend_samples.append(sample)
            attempts += 1
        if len(valid_spend_samples) < n_samples:
            print(f"WARNING: Only {len(valid_spend_samples)} valid samples found after {attempts} attempts (target was {n_samples}).")
        spend_grid = np.array(valid_spend_samples)
        incremental_outcome_grid = np.zeros(len(spend_grid))
        # print(f"Spend grid shape: {spend_grid.shape}")
        # print(f"Incremental outcome grid shape: {incremental_outcome_grid.shape}")

        original_media = self._meridian.input_data.media
        original_media_spend = self._meridian.input_data.media_spend
        original_organic_media = self._meridian.input_data.organic_media
        original_non_media_treatments = self._meridian.input_data.non_media_treatments
        original_controls = self._meridian.input_data.controls
        
        # Get the selected geo index
        selected_geo_idx = self._meridian.input_data.geo.values.tolist().index(self.selected_geo)
        
        # Create boolean mask for selected times
        all_times = original_media.media_time.values
        time_mask = np.isin(all_times, selected_times)
        
        # Get geo-specific data for the selected period
        geo_specific_media = original_media.isel(geo=selected_geo_idx, media_time=time_mask)
        geo_specific_spend = original_media_spend.isel(geo=selected_geo_idx, time=time_mask)

        # Calculate the proportion of spend for each time slice (pre-compute once)
        geo_specific_spend_tf = tf.convert_to_tensor(geo_specific_spend, dtype=tf.float32)
        geo_specific_media_tf = tf.convert_to_tensor(geo_specific_media, dtype=tf.float32)
        time_proportions = tf.math.divide_no_nan(
            geo_specific_spend_tf,
            tf.math.reduce_sum(geo_specific_spend_tf, axis=0)
        )
        
        # Get shape information for efficient indexing
        n_geos, n_times, n_channels = original_media.values.shape
        
        # Process samples in batches to reduce overhead and improve TensorFlow utilization
        # This optimization:
        # 1. Reduces TensorFlow operation overhead by batching computations
        # 2. Improves memory locality and cache efficiency
        # 3. Allows TensorFlow to optimize operations across multiple samples
        # Use a reasonable batch size (not too large to avoid memory issues)
        sample_batch_size = min(20, max(5, n_samples // 10))  # Adaptive batch size
        
        # Fill in the grids using batched processing
        for batch_start in range(0, n_samples, sample_batch_size):
            batch_end = min(batch_start + sample_batch_size, n_samples)
            batch_size_actual = batch_end - batch_start
            batch_spend = spend_grid[batch_start:batch_end]  # Shape: (batch_size, n_channels)
            
            # Vectorized computation for the entire batch
            # Expand dimensions: (batch_size, 1, n_channels) * (1, n_times, n_channels) -> (batch_size, n_times, n_channels)
            batch_spend_tf = tf.convert_to_tensor(batch_spend, dtype=tf.float32)  # (batch_size, n_channels)
            batch_spend_expanded = tf.expand_dims(batch_spend_tf, axis=1)  # (batch_size, 1, n_channels)
            time_proportions_expanded = tf.expand_dims(time_proportions, axis=0)  # (1, n_times, n_channels)
            
            # Apply time proportions to all samples in batch
            batch_time_sliced_spend = batch_spend_expanded * time_proportions_expanded  # (batch_size, n_times, n_channels)
            
            # Calculate scaling factors for the batch
            geo_specific_spend_expanded = tf.expand_dims(geo_specific_spend_tf, axis=0)  # (1, n_times, n_channels)
            batch_scaling_factors = tf.math.divide_no_nan(
                batch_time_sliced_spend,
                geo_specific_spend_expanded
            )  # (batch_size, n_times, n_channels)
            
            # Scale media values for the batch
            geo_specific_media_expanded = tf.expand_dims(geo_specific_media_tf, axis=0)  # (1, n_times, n_channels)
            batch_scaled_media = batch_scaling_factors * geo_specific_media_expanded  # (batch_size, n_times, n_channels)
            
            # Process each sample in the batch (incremental_outcome doesn't support batching across samples)
            # Pre-allocate arrays for the batch to reduce memory allocation overhead
            batch_scaled_media_np = batch_scaled_media.numpy()  # Convert entire batch to numpy once
            
            for batch_idx in range(batch_size_actual):
                i = batch_start + batch_idx
                
                # Get the scaled media for this sample
                scaled_media_sample = batch_scaled_media_np[batch_idx]  # (n_times, n_channels)
                
                # Create modified versions (reuse original structure, only modify selected slice)
                # Use numpy views/assignments to avoid full copy when possible
                new_media_values = original_media.values.copy()
                new_media_spend_values = original_media_spend.values.copy()
                
                # Replace the selected time slice on the selected geo with scaled media values
                new_media_values[selected_geo_idx, time_mask, :] = scaled_media_sample
                new_media_spend_values[selected_geo_idx, time_mask, :] = geo_specific_spend.values
                
                # Create xarray DataArrays from the modified values (xarray requires copy for safety)
                new_media = original_media.copy()
                new_media.values = new_media_values
                new_media_spend = original_media_spend.copy()
                new_media_spend.values = new_media_spend_values
                new_organic_media = original_organic_media.copy() if original_organic_media is not None else None
                
                # Calculate incremental outcome using analyzer with proper geo filtering
                incremental_outcome = self._analyzer.incremental_outcome(
                    use_posterior=use_posterior,
                    new_data=analyzer.DataTensors(
                        media_spend=tf.cast(new_media_spend, tf.float32),
                        media=tf.cast(new_media, tf.float32),
                        organic_media=tf.cast(new_organic_media, tf.float32) if new_organic_media is not None else None,
                    ),
                    selected_geos=[self.selected_geo],
                    selected_times=selected_times,
                    use_kpi=use_kpi,
                    include_non_paid_channels=False,
                    batch_size=batch_size,
                )
                # Convert to numpy array if it's a tensor
                if isinstance(incremental_outcome, tf.Tensor):
                    incremental_outcome = incremental_outcome.numpy()
                # Calculate total across all dimensions to get the total outcome
                incremental_outcome_grid[i] = np.sum(incremental_outcome)
        
        return spend_grid, incremental_outcome_grid

    def _create_grid_dataset(
        self,
        spend_grid: np.ndarray,
        spend_step_size: float,
        incremental_outcome_grid: np.ndarray,
    ) -> xr.Dataset:
        """Creates a dataset from the optimization grid.

        Args:
            spend_grid: The spend grid array.
            spend_step_size: The step size used for the spend grid.
            incremental_outcome_grid: The incremental outcome grid array.

        Returns:
            An xarray Dataset containing the spend and incremental outcome grids.
        """
        n_grid_points, n_channels = spend_grid.shape
        
        # Create coordinates for the grid
        grid_point_coords = np.arange(n_grid_points)
        channel_coords = np.arange(n_channels)
        
        # Create the dataset
        dataset = xr.Dataset(
            {
                "spend": (["grid_point", "channel"], spend_grid),
                "incremental_outcome": (["grid_point"], incremental_outcome_grid),
            },
            coords={
                "grid_point": grid_point_coords,
                "channel": channel_coords,
            },
            attrs={
                "spend_step_size": spend_step_size,
            },
        )
        
        return dataset

    def _create_optimization_results(
        self,
        spend: np.ndarray,
        selected_times: Sequence[str] | None = None,
        selected_geo: str | None = None,
        use_posterior: bool = True,
        use_kpi: bool = False,
        use_optimal_frequency: bool = False,
        optimal_frequency: xr.DataArray | None = None,
        batch_size: int = c.DEFAULT_BATCH_SIZE,
    ) -> xr.Dataset:
        """Creates a dataset with optimization results.

        Args:
            spend: The spend allocation to evaluate.
            selected_times: Selected time points.
            selected_geo: Selected geography.
            use_posterior: Whether to use posterior distributions.
            use_kpi: Whether to use KPI instead of revenue.
            use_optimal_frequency: Whether to use optimal frequency.
            optimal_frequency: Optional optimal frequency values.
            batch_size: Batch size for computations.

        Returns:
            An xarray Dataset containing the optimization results.
        """
        # Get valid geo names from the model's input data
        valid_geos = self._meridian.input_data.geo.values.tolist()
        
        # If a geo is selected, ensure it's valid
        selected_geos = None
        if selected_geo:
            if selected_geo not in valid_geos:
                raise ValueError(
                    f"Invalid geo '{selected_geo}'. Valid geos are: {valid_geos}"
                )
            selected_geos = [selected_geo]
            selected_geo_idx = valid_geos.index(selected_geo)
        
        # Get the channel names from the model
        channel_names = self._meridian.input_data.media_channel.values.tolist()
        
        # Get original data
        original_media = self._meridian.input_data.media
        original_organic_media = self._meridian.input_data.organic_media
        original_media_spend = self._meridian.input_data.media_spend
        original_non_media_treatments = self._meridian.input_data.non_media_treatments
        original_controls = self._meridian.input_data.controls
        # Create boolean mask for selected times
        all_times = original_media.media_time.values
        time_mask = np.isin(all_times, selected_times)
        
        # Calculate the geo-specific media values
        geo_specific_media = original_media.isel(geo=selected_geo_idx, media_time=time_mask)
        geo_specific_spend = original_media_spend.isel(geo=selected_geo_idx, time=time_mask)
        
        # Calculate the spend ratio
        # Check if shapes match before reshaping
        if spend.shape != geo_specific_spend.shape:
            # Reshape spend to match geo_specific_spend shape while preserving total
            spend_reshaped = np.tile(spend, (geo_specific_spend.shape[0], 1))
            spend_reshaped = spend_reshaped * (np.sum(spend) / np.sum(spend_reshaped))
            spend_to_use = spend_reshaped
        else:
            spend_to_use = spend
         # it's not fair in case of optimized spend, as it's going to ignore non-0 spend_to_use values where geo_specific_spend is 0   
        spend_ratio = tf.math.divide_no_nan(
            tf.convert_to_tensor(spend_to_use, dtype=tf.float32),
            tf.convert_to_tensor(geo_specific_spend, dtype=tf.float32)
        )
        if spend.ndim > 1: # if it's the non-optimized spend
            scaling_factor = tf.math.divide_no_nan(
            tf.convert_to_tensor(spend, dtype=tf.float32),
            tf.convert_to_tensor(geo_specific_spend, dtype=tf.float32)
        )
            geo_specific_media_scaled3 = scaling_factor * geo_specific_media
        else:
            # Calculate the proportion of spend for each time slice
            time_proportions = tf.math.divide_no_nan(
                tf.convert_to_tensor(geo_specific_spend, dtype=tf.float32),
                tf.math.reduce_sum(tf.convert_to_tensor(geo_specific_spend, dtype=tf.float32), axis=0)
            )
            
            # Apply these proportions to the single-channel spend
            time_sliced_spend = tf.convert_to_tensor(spend, dtype=tf.float32) * time_proportions
            
            # Calculate scaling factor for each time slice
            scaling_factor = tf.math.divide_no_nan(
                time_sliced_spend,
                tf.convert_to_tensor(geo_specific_spend, dtype=tf.float32)
            )
            
            # Scale the media values using the time-sliced scaling factors
            geo_specific_media_scaled3 = scaling_factor * geo_specific_media

        # Create modified versions
        new_media = original_media.copy()
        new_media_spend = original_media_spend.copy()
        
        # Replace the selected time slice on the selected geo with geo_specific_media values
        new_media.values[selected_geo_idx, time_mask, :] = geo_specific_media_scaled3 # geo_specific_media_scaled.numpy()
        new_media_spend.values[selected_geo_idx, time_mask, :] = spend_to_use #geo_specific_spend.values # TODO: it's just using the original values
        # although it's probably not a problem, because _analyzer.incremental_outcome() is not using the spend values but media values
        new_organic_media = original_organic_media.copy() if original_organic_media is not None else None

        # Calculate incremental outcome using analyzer with proper geo filtering
        incremental_outcome = self._analyzer.incremental_outcome(
            use_posterior=use_posterior,
            new_data=analyzer.DataTensors(
                media_spend=tf.cast(new_media_spend, tf.float32),
                media=tf.cast(new_media, tf.float32),
                organic_media=tf.cast(new_organic_media, tf.float32) if new_organic_media is not None else None,
            ),
            selected_geos=selected_geos,
            selected_times=selected_times,
            use_kpi=use_kpi,
            include_non_paid_channels=False,
            batch_size=batch_size,
        )

        # Convert to numpy array if it's a tensor
        if isinstance(incremental_outcome, tf.Tensor):
            incremental_outcome = incremental_outcome.numpy()

        # Calculate mean across posterior samples and draws, but preserve channel dimension
        if incremental_outcome.ndim > 2:  # If we have chains and draws dimensions
            incremental_outcome = np.mean(incremental_outcome, axis=tuple(range(incremental_outcome.ndim - 1)), dtype=np.float64)
        
        # Ensure incremental_outcome is 1D (channels)
        if incremental_outcome.ndim > 1:
            incremental_outcome = np.sum(incremental_outcome, axis=tuple(range(incremental_outcome.ndim - 1)))
        
        # Calculate ROI and mROI
        total_spend = np.sum(spend)

        # Ensure spend is 1D (channels)
        if spend.ndim != 1: # if it's the non-optimized spend
            spend_output = np.sum(spend, axis=tuple(range(spend.ndim - 1)))
        else:
            spend_output = spend

        # Calculate ROI elementwise, handling division by zero safely
        roi = np.zeros_like(incremental_outcome, dtype=np.float64)
        mask = spend_output > 0
        roi[mask] = incremental_outcome[mask] / spend_output[mask]
        mroi = roi  # Using ROI as a proxy for mROI in this simplified version

        # Calculate pct_of_spend and pct_of_contribution
        pct_of_spend = spend_output / total_spend if total_spend > 0 else 0
        total_outcome = np.sum(incremental_outcome)
        pct_of_contribution = incremental_outcome / total_outcome if total_outcome > 0 else 0

        # Calculate effectiveness and cpik
        effectiveness = incremental_outcome / spend_output if np.any(spend_output > 0) else 0
        cpik = spend_output / incremental_outcome if np.any(incremental_outcome > 0) else 0

        # Create the dataset with all metrics
        dataset = xr.Dataset(
            {
                "spend": (["channel"], spend_output),
                "incremental_outcome": (["channel"], incremental_outcome),
                "pct_of_spend": (["channel"], pct_of_spend),
                "pct_of_contribution": (["channel"], pct_of_contribution),
                "effectiveness": (["channel"], effectiveness),
                "roi": (["channel"], roi),
                "mroi": (["channel"], mroi),
                "cpik": (["channel"], cpik),
            },
            coords={
                "channel": channel_names
            },
            attrs={
                "total_spend": total_spend,
                "use_posterior": use_posterior,
                "use_kpi": use_kpi,
                "use_optimal_frequency": use_optimal_frequency,
            },
        )
        
        return dataset

    def test_incremental_outcome(self, reference_period: tuple[str, str]|None=None):
        """Test function to compare incremental outcomes for different media patterns."""
        print("\n=== Starting Incremental Outcome Test ===")
        
        # Use base Analyzer instead of GeoAnalyzer
        from meridian.analysis import analyzer
        base_analyzer = analyzer.Analyzer(self._meridian)
        
        # Get original data
        original_media = self._meridian.input_data.media
        original_organic_media = self._meridian.input_data.organic_media
        original_media_spend = self._meridian.input_data.media_spend
        original_non_media_treatments = self._meridian.input_data.non_media_treatments
        original_controls = self._meridian.input_data.controls
        
        # Create modified versions
        modified_media = original_media.copy()
        modified_media.values = np.ones_like(original_media.values)
        modified_organic_media = original_organic_media.copy() if original_organic_media is not None else None
        # non_media_treatments and controls are ignored in the analyzer:
        # modified_non_media_treatments = original_non_media_treatments.copy() if original_non_media_treatments is not None else None
        # modified_controls = original_controls.copy() if original_controls is not None else None
        modified_organic_media.values = np.ones_like(original_organic_media.values)
        modified_media_spend = original_media_spend.copy()
        modified_media_spend.values = np.ones_like(original_media_spend.values)
        
        # Test 1: Original data
        print("\nTest 1: Original data")
        outcome1 = base_analyzer.incremental_outcome(
            use_posterior=True,
            new_data=analyzer.DataTensors(
                media_spend=tf.cast(original_media_spend, tf.float32),
                media=tf.cast(original_media, tf.float32),
                organic_media=tf.cast(original_organic_media, tf.float32) if original_organic_media is not None else None,
            ),
            selected_times=reference_period,
            use_kpi=True,
            include_non_paid_channels=False,
            batch_size=32
        )
        
        # Test 2: Modified media spend
        print("\nTest 2: Modified media spend")
        outcome2 = base_analyzer.incremental_outcome(
            use_posterior=True,
            new_data=analyzer.DataTensors(
                media_spend=tf.cast(modified_media_spend, tf.float32),
                media=tf.cast(original_media, tf.float32),
                organic_media=tf.cast(original_organic_media, tf.float32) if original_organic_media is not None else None
            ),
            selected_times=reference_period,
            use_kpi=True,
            include_non_paid_channels=False,
            batch_size=32
        )
        
        # Test 3: Modified media
        print("\nTest 3: Modified media")
        outcome3 = base_analyzer.incremental_outcome(
            use_posterior=True,
            new_data=analyzer.DataTensors(
                media_spend=tf.cast(original_media_spend, tf.float32),
                media=tf.cast(modified_media, tf.float32),
                organic_media=tf.cast(original_organic_media, tf.float32) if original_organic_media is not None else None
            ),
            selected_times=reference_period,
            use_kpi=True,
            include_non_paid_channels=False,
            batch_size=32
        )
        
        # Test 4: Modified organic media
        print("\nTest 4: Modified organic media")
        outcome4 = base_analyzer.incremental_outcome(
            use_posterior=True,
            new_data=analyzer.DataTensors(
                media_spend=tf.cast(original_media_spend, tf.float32),
                media=tf.cast(original_media, tf.float32),
                organic_media=tf.cast(modified_organic_media, tf.float32) if modified_organic_media is not None else None
            ),
            selected_times=reference_period,
            use_kpi=True,
            include_non_paid_channels=False,
            batch_size=32
        )
        
        # Convert outcomes to numpy arrays if they're tensors
        outcomes = []
        for i, outcome in enumerate([outcome1, outcome2, outcome3, outcome4]):
            if isinstance(outcome, tf.Tensor):
                outcome = outcome.numpy()
            if outcome.ndim > 2:
                outcome = np.mean(outcome, axis=tuple(range(outcome.ndim - 1)))
            outcomes.append(outcome)
            print(f"\nOutcome {i+1} shape: {outcome.shape}")
            print(f"Outcome {i+1}: {outcome}")
        
        # Compare results
        print("\nComparing results:")
        for i in range(len(outcomes)):
            for j in range(i+1, len(outcomes)):
                diff = outcomes[i] - outcomes[j]
                print(f"\nDifference between test {i+1} and test {j+1}:")
                print(f"Absolute difference: {diff}")
                max_val = np.maximum(np.abs(outcomes[i]), np.abs(outcomes[j]))
                relative_diff = np.where(max_val != 0, diff / max_val * 100, 0)
                print(f"Relative difference (%): {relative_diff}")
                print(f"Mean relative difference: {np.mean(relative_diff)}%")
                print(f"Max relative difference: {np.max(relative_diff)}%")
        
        return outcomes


def _optimize_single_geo_worker(args):
    """Worker function for single geo optimization using multiprocessing.
    
    This function must be at module level to be picklable for multiprocessing.
    """
    # Suppress warnings in worker process
    import warnings
    import os
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Configure TensorFlow in worker process to use fewer threads per process
    # This prevents processes from competing for CPU resources
    try:
        import multiprocessing as mp
        total_cores = mp.cpu_count()
        # Use fewer threads per process to avoid over-subscription
        # If we have N processes, each should use ~total_cores/N threads
        threads_per_process = max(1, total_cores // 4)  # Conservative: use 1/4 of cores per process
        tf.config.threading.set_intra_op_parallelism_threads(threads_per_process)
        tf.config.threading.set_inter_op_parallelism_threads(threads_per_process)
    except Exception:
        pass  # Ignore if configuration fails
    
    (geo_name, meridian_model, optimization_date_range, geo_opt_dict,
     spend_constraint_lower, spend_constraint_upper, n_samples,
     use_kpi, use_posterior, use_optimal_frequency, batch_size, verbose) = args
    
    try:
        # Recreate optimizer in worker process (avoids pickling issues)
        worker_optimizer = BudgetOptimizerGeo(meridian_model)
        
        # Get geo-specific parameters
        geo_params = geo_opt_dict.get(geo_name, {})
        total_spend = geo_params.get('total_spend', 0)
        step_size = geo_params.get('step_size', 1000)
        
        if total_spend <= 0:
            print(f"Warning: {geo_name} has zero or negative total spend. Skipping.")
            return geo_name, None
        
        # Run optimization
        results = worker_optimizer.optimize(
            selected_times=optimization_date_range,
            selected_geo=geo_name,
            fixed_budget=True,
            budget=total_spend,
            spend_constraint_lower=spend_constraint_lower,
            spend_constraint_upper=spend_constraint_upper,
            step_size=step_size,
            n_samples=n_samples,
            use_kpi=use_kpi,
            use_posterior=use_posterior,
            use_optimal_frequency=use_optimal_frequency,
            batch_size=batch_size,
            verbose=verbose,
        )
        
        print(f"Completed optimization for {geo_name}")
        return geo_name, results
        
    except Exception as e:
        print(f"Error optimizing {geo_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return geo_name, None


def _optimize_geos_sequential(
    budget_optimizer_geo: BudgetOptimizerGeo,
    geos_to_optimize: list[str],
    optimization_date_range: tuple[str, str],
    geo_opt_dict: dict,
    spend_constraint_lower: float,
    spend_constraint_upper: float,
    n_samples: int,
    use_kpi: bool,
    use_posterior: bool,
    use_optimal_frequency: bool,
    batch_size: int,
    verbose: bool,
) -> dict:
    """Optimize geos sequentially with TensorFlow parallelism."""
    if verbose:
        print(f"Using sequential optimization with TensorFlow parallelism for {len(geos_to_optimize)} geos...")
    
    optimization_results = {}
    for geo_name in geos_to_optimize:
        try:
            geo_params = geo_opt_dict.get(geo_name, {})
            total_spend = geo_params.get('total_spend', 0)
            step_size = geo_params.get('step_size', 1000)
            
            if total_spend <= 0:
                print(f"Warning: {geo_name} has zero or negative total spend. Skipping.")
                continue
            
            results = budget_optimizer_geo.optimize(
                selected_times=optimization_date_range,
                selected_geo=geo_name,
                fixed_budget=True,
                budget=total_spend,
                spend_constraint_lower=spend_constraint_lower,
                spend_constraint_upper=spend_constraint_upper,
                step_size=step_size,
                n_samples=n_samples,
                use_kpi=use_kpi,
                use_posterior=use_posterior,
                use_optimal_frequency=use_optimal_frequency,
                batch_size=batch_size,
                verbose=verbose,
            )
            optimization_results[geo_name] = results
            if verbose:
                print(f"Completed optimization for {geo_name}")
        except Exception as e:
            print(f"Error optimizing {geo_name}: {str(e)}")
    
    print(f"Completed optimization. Successfully optimized {len(optimization_results)} out of {len(geos_to_optimize)} geos.")
    return optimization_results


def _optimize_geos_with_multiprocessing(
    meridian_model,
    geos_to_optimize: list[str],
    optimization_date_range: tuple[str, str],
    geo_opt_dict: dict,
    spend_constraint_lower: float,
    spend_constraint_upper: float,
    n_samples: int,
    n_processes: int,
    use_kpi: bool,
    use_posterior: bool,
    use_optimal_frequency: bool,
    batch_size: int,
    verbose: bool,
) -> dict:
    """Optimize geos using multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp

    worker_args = [
        (geo_name, meridian_model, optimization_date_range, geo_opt_dict,
         spend_constraint_lower, spend_constraint_upper, n_samples,
         use_kpi, use_posterior, use_optimal_frequency, batch_size, verbose)
        for geo_name in geos_to_optimize
    ]
    
    print(f"Starting parallel optimization for {len(geos_to_optimize)} geos using {n_processes} processes...")
    
    optimization_results = {}
    successful_geos = 0
    # Force 'spawn' context to avoid TensorFlow deadlocks on Linux
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(
        max_workers=n_processes, 
        mp_context=ctx) as executor:
        future_to_geo = {
            executor.submit(_optimize_single_geo_worker, args): args[0]
            for args in worker_args
        }
        
        for future in as_completed(future_to_geo):
            geo_name = future_to_geo[future]
            try:
                geo_name, result = future.result()
                if result is not None:
                    optimization_results[geo_name] = result
                    successful_geos += 1
                else:
                    print(f"Skipping {geo_name} due to optimization failure")
            except Exception as e:
                print(f"Exception occurred while processing {geo_name}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"Completed parallel optimization. Successfully optimized {successful_geos} out of {len(geos_to_optimize)} geos.")
    return optimization_results


def _optimize_geos_with_threading(
    budget_optimizer_geo: BudgetOptimizerGeo,
    geos_to_optimize: list[str],
    optimization_date_range: tuple[str, str],
    geo_opt_dict: dict,
    spend_constraint_lower: float,
    spend_constraint_upper: float,
    n_samples: int,
    n_processes: int,
    use_kpi: bool,
    use_posterior: bool,
    use_optimal_frequency: bool,
    batch_size: int,
    verbose: bool,
) -> dict:
    """Optimize geos using threading (fallback when multiprocessing unavailable)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def optimize_single_geo_worker_thread(geo_name):
        """Worker function for single geo optimization using threading."""
        try:
            geo_params = geo_opt_dict.get(geo_name, {})
            total_spend = geo_params.get('total_spend', 0)
            step_size = geo_params.get('step_size', 1000)
            
            if total_spend <= 0:
                print(f"Warning: {geo_name} has zero or negative total spend. Skipping.")
                return geo_name, None
            
            results = budget_optimizer_geo.optimize(
                selected_times=optimization_date_range,
                selected_geo=geo_name,
                fixed_budget=True,
                budget=total_spend,
                spend_constraint_lower=spend_constraint_lower,
                spend_constraint_upper=spend_constraint_upper,
                step_size=step_size,
                n_samples=n_samples,
                use_kpi=use_kpi,
                use_posterior=use_posterior,
                use_optimal_frequency=use_optimal_frequency,
                batch_size=batch_size,
                verbose=verbose,
            )
            
            print(f"Completed optimization for {geo_name}")
            return geo_name, results
            
        except Exception as e:
            print(f"Error optimizing {geo_name}: {str(e)}")
            return geo_name, None
    
    print(f"Starting parallel optimization for {len(geos_to_optimize)} geos using {n_processes} threads...")
    
    optimization_results = {}
    successful_geos = 0
    
    with ThreadPoolExecutor(max_workers=n_processes) as executor:
        future_to_geo = {
            executor.submit(optimize_single_geo_worker_thread, geo_name): geo_name
            for geo_name in geos_to_optimize
        }
        
        for future in as_completed(future_to_geo):
            geo_name = future_to_geo[future]
            try:
                geo_name, result = future.result()
                if result is not None:
                    optimization_results[geo_name] = result
                    successful_geos += 1
                else:
                    print(f"Skipping {geo_name} due to optimization failure")
            except Exception as e:
                print(f"Exception occurred while processing {geo_name}: {str(e)}")
    
    print(f"Completed parallel optimization. Successfully optimized {successful_geos} out of {len(geos_to_optimize)} geos.")
    return optimization_results


def optimize_geos_parallel(
    budget_optimizer_geo: BudgetOptimizerGeo,
    geos_to_optimize: list[str],
    optimization_date_range: tuple[str, str],
    geo_opt_dict: dict,
    spend_constraint_lower: float = 0.5,
    spend_constraint_upper: float = 0.5,
    n_samples: int = 500,
    n_processes: int | None = None,
    use_kpi: bool = False,
    use_posterior: bool = True,
    use_optimal_frequency: bool = True,
    batch_size: int = 32,
    verbose: bool = False,
    use_multiprocessing: bool | None = None,  # None = auto-detect, True = force, False = disable
) -> dict:
    """
    Optimize multiple geos in parallel using multiprocessing (bypasses GIL for true parallelism).
    
    Note: This function uses multiprocessing which requires pickling the optimizer. If pickling fails,
    the function will fall back to threading (which has GIL limitations).
    
    For small workloads (< 4 geos), a simple sequential loop with TensorFlow parallelism is often faster
    due to multiprocessing overhead. The function auto-detects this unless use_multiprocessing is set.
    
    Args:
        budget_optimizer_geo: The BudgetOptimizerGeo instance to use for optimization.
        geos_to_optimize: List of geo names to optimize.
        optimization_date_range: Tuple of (start_date, end_date) for optimization period.
        geo_opt_dict: Dictionary containing geo-specific optimization parameters.
        spend_constraint_lower: Lower spend constraint (default: 0.5).
        spend_constraint_upper: Upper spend constraint (default: 0.5).
        n_samples: Number of samples for optimization grid (default: 500).
        n_processes: Number of processes to use. If None, uses CPU count / 2 to avoid over-subscription.
        use_kpi: Whether to use KPI instead of revenue (default: False).
        use_posterior: Whether to use posterior distributions (default: True).
        use_optimal_frequency: Whether to use optimal frequency (default: True).
        batch_size: Batch size for computations (default: 32).
        verbose: Whether to print verbose output (default: False).
        use_multiprocessing: Whether to use multiprocessing. None = auto-detect (use for 4+ geos),
            True = force multiprocessing, False = use sequential loop with TensorFlow parallelism.
    
    Returns:
        Dictionary mapping geo names to their optimization results.
    """
    import multiprocessing as mp
    import pickle
    
    # Auto-detect whether to use multiprocessing based on workload size
    if use_multiprocessing is None:
        use_multiprocessing = len(geos_to_optimize) >= 4  # Threshold: use multiprocessing for 4+ geos
    
    # If multiprocessing is disabled, use sequential loop
    if not use_multiprocessing:
        return _optimize_geos_sequential(
            budget_optimizer_geo, geos_to_optimize, optimization_date_range, geo_opt_dict,
            spend_constraint_lower, spend_constraint_upper, n_samples,
            use_kpi, use_posterior, use_optimal_frequency, batch_size, verbose
        )
    
    # Calculate optimal number of processes
    if n_processes is None:
        cpu_count = mp.cpu_count()
        n_processes = min(max(1, cpu_count // 2), len(geos_to_optimize))
    
    # Test if model can be pickled for multiprocessing
    meridian_model = budget_optimizer_geo._meridian
    can_use_multiprocessing = False
    try:
        pickle.dumps(meridian_model)
        can_use_multiprocessing = True
    except (pickle.PicklingError, AttributeError, TypeError) as e:
        print(f"⚠️  Multiprocessing not available: Model cannot be pickled ({type(e).__name__}: {str(e)})")
        print(f"⚠️  Falling back to threading (limited by Python GIL - may not provide significant speedup)")
        print(f"⚠️  For true parallelism, consider using a model serialization approach or optimizing the inner loop")
    
    # Try multiprocessing, fall back to threading on failure
    if can_use_multiprocessing:
        try:
            return _optimize_geos_with_multiprocessing(
                meridian_model, geos_to_optimize, optimization_date_range, geo_opt_dict,
                spend_constraint_lower, spend_constraint_upper, n_samples, n_processes,
                use_kpi, use_posterior, use_optimal_frequency, batch_size, verbose
            )
        except Exception as e:
            print(f"⚠️  Multiprocessing failed: {str(e)}. Falling back to threading...")
            import traceback
            traceback.print_exc()
    
    # Fall back to threading
    return _optimize_geos_with_threading(
        budget_optimizer_geo, geos_to_optimize, optimization_date_range, geo_opt_dict,
        spend_constraint_lower, spend_constraint_upper, n_samples, n_processes,
        use_kpi, use_posterior, use_optimal_frequency, batch_size, verbose
    )

def worker_optimization_task(args):
    """
    Runs optimization for a single geo inside a separate process.
    """
    (geo_name, meridian_model, date_range, lower, upper, verbose) = args

    try:
        # 2. Use 'optimizer' directly (matches the import above)
        local_optimizer = optimizer.BudgetOptimizer(meridian_model)
        
        start_date, end_date = date_range
        
        result = local_optimizer.optimize(
            start_date=start_date,
            end_date=end_date,
            selected_geos=[geo_name],
            fixed_budget=True,
            spend_constraint_lower=lower,
            spend_constraint_upper=upper
        )
        
        if verbose:
            print(f"✅ [Process {mp.current_process().name}] Finished {geo_name}")
            
        return geo_name, result

    except Exception as e:
        print(f"❌ [Process {mp.current_process().name}] Error on {geo_name}: {e}")
        return geo_name, None