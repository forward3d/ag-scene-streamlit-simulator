"""Scene MX package for marketing mix modeling and optimization."""

from . import geo_optimizer
from .geo_optimizer import (
    BudgetOptimizerGeo,
    optimize_geos_parallel,
    worker_optimization_task
)
from .export_data import *
from .utils import *

__version__ = "1.0.1"
# Backwards-compatible alias (some code uses `scene_mx.version`)
version = __version__

# Avoid importing TensorFlow-heavy modules at package import time.
# Users can still access `scene_mx.modelling` after `import scene_mx` via lazy import.
def __getattr__(name: str):
    if name == "modelling":
        import importlib
        return importlib.import_module(".modelling", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["modelling"])


__all__ = [
    'geo_optimizer', 
    'BudgetOptimizerGeo',
    'optimize_geos_parallel',
    'modelling',
    'get_optimized_df', 
    'get_summary_df', 
    'dataset_to_dataframe', 
    'get_response_curves_df', 
    'export_results', 
    'upload_csv_to_gcs',
    'upload_to_bq',
    'create_bq_schema',
    'save_mmm',
    # Utils functions
    'print_gcs_filenames',
    'get_latest_data_file',
    'save_mmm_to_gcs',
    'load_pickle_from_gcs',
    'extract_roi_from_trained_model',
    'get_variable_names',
    'build_opt_constraints',
    'export_parameters_to_yaml',
    'get_model_spec_attrs',
    'get_variable_type',
    'worker_optimization_task'
]
