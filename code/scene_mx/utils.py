"""Utility functions for GCS operations, model I/O, and data processing.

This module provides helper functions for:
- Google Cloud Storage (GCS) operations (listing files, loading/saving models)
- Model I/O operations (pickle serialization, MMM model handling)
- ROI extraction from trained models
- Variable name extraction from config files or dataframes
- Export model parameters and ROI priors to a YAML file
"""

import pickle
from google.cloud import storage
import tempfile
import numpy as np
import pandas as pd
import copy
import yaml
import os

# Print all files in a GCS bucket


def print_gcs_filenames(bucket_name, prefix=None):
    """
    Print all files in a GCS bucket
    Args:
        bucket_name: The name of the GCS bucket
        prefix: The prefix to the data in the GCS bucket
    Returns:
        None
    """
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    print(f"Listing blobs in bucket {bucket_name} with prefix {prefix}")
    blobs = bucket.list_blobs(prefix=prefix) if prefix else bucket.list_blobs()
    # print(f"Found {len(blobs)} blobs")
    for blob in blobs:
        print(blob.name)

# Example
# print_gcs_filenames(BUCKET_NAME, prefix = "scene/model/')

# Get the latest data file from a GCS bucket
from google.cloud import storage
from datetime import datetime

def get_latest_data_file(bucket_name, path):
    """
    Get the latest data file from a GCS bucket
    Args:
        bucket_name: The name of the GCS bucket
        path: The path to the data in the GCS bucket
    Returns:
        The name of the latest data file and the update time
    """
    print(f"Getting the latest data file from {bucket_name} in {path}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=path)
    latest_blob = None
    latest_time = None
    for blob in blobs:
        if latest_time is None or blob.updated > latest_time:
            latest_time = blob.updated
            latest_blob = blob
    
    if latest_blob:
        return latest_blob.name, latest_blob.updated
    
    return None, None

# Example usage
# latest_file, update_time = get_latest_data_file(BUCKET_NAME, PATH)
# if latest_file:
#     print(f"Latest data file: {latest_file}")
#     print(f"Last modified: {update_time}")
# else:
#     print("No data files found")



def save_mmm_to_gcs(mmm_model, bucket_name, blob_path):
    """
    Save a MMM model to a GCS bucket
    Args:
        mmm_model: The MMM model to save
        bucket_name: The name of the GCS bucket
        blob_path: The path to the pickle file in the GCS bucket
    Returns:
        None
    """
    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile() as tmp:
        pickle.dump(mmm_model, tmp)
        tmp.flush()
        tmp.seek(0)
        # Upload to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_file(tmp)
    print(f"MMM model written to gs://{bucket_name}/{blob_path}")

# Example usage:
# save_mmm_to_gcs(mmm, BUCKET_NAME, MODEL_PATH)

def load_pickle_from_gcs(bucket_name, blob_path):
    """
    Load a pickle file from a GCS bucket
    Args:
        bucket_name: The name of the GCS bucket
        blob_path: The path to the pickle file in the GCS bucket
    Returns:
        The loaded pickle file
    """
    from google.cloud import storage
    import pickle
    import tempfile
    from meridian.model import model

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    # Download the blob's contents as bytes to a temporary file and load pickle
    with tempfile.NamedTemporaryFile(delete=False) as temp_pkl_file:
        blob.download_to_filename(temp_pkl_file.name)
    mmm = model.load_mmm(temp_pkl_file.name)
    print(f"MMM model loaded from {blob.name}, base model exists and will be loaded.")
    return mmm

# Example usage:
# model_fname_prev = 'AMAN_hierarchical_mmm_20251105.pkl'
# previous_mmm = load_pickle_from_gcs(BUCKET_NAME, f'scene/model/{model_fname_prev}')


# If you have a trained model loaded, you can extract its ROI posterior statistics
# to use them as priors for a new model training

# def extract_roi_from_trained_model(trained_mmm, use_posterior_mean=True, sigma_multiplier=1.0, assume_log_space=False):
#     """
#     Extract ROI statistics from a trained model and convert to roi_m_dictionary format.
    
#     IMPORTANT: Since LogNormal priors require loc and scale parameters in log space,
#     this function transforms ROI values from original scale to log space.
    
#     Args:
#         trained_mmm: A trained Meridian model object
#         use_posterior_mean: If True, use posterior mean as ROI estimate. If False, use median.
#         sigma_multiplier: Multiplier for standard deviation (default 1.0). 
#                           Use < 1.0 to tighten priors, > 1.0 to loosen them.
#         assume_log_space: If True, assumes roi_m posterior samples are already in log space.
#                           If False (default), assumes they're in original scale and transforms them.
    
#     Returns:
#         Dictionary in format: {'channel_name': (roi_loc, roi_scale)} where loc and scale
#         are in log space (ready for LogNormal distribution).
#     """
#     # Get ROI posterior samples
#     roi_samples = trained_mmm.inference_data.posterior['roi_m'].values
    
#     # Get channel names
#     channel_names = trained_mmm.input_data.media_channel.values
    
#     # Calculate statistics
#     roi_dict = {}
    
#     for i, channel in enumerate(channel_names):
#         # Get posterior samples for this channel
#         channel_samples = roi_samples[:, :, i].flatten()  # Flatten across chains and draws
        
#         # Transform to log space if samples are in original scale
#         if not assume_log_space:
#             # Ensure all values are positive before taking log
#             channel_samples = np.maximum(channel_samples, 1e-6)
#             channel_samples_log = np.log(channel_samples)
#         else:
#             channel_samples_log = channel_samples
        
#         # Calculate mean and std in log space (for LogNormal loc and scale parameters)
#         if use_posterior_mean:
#             roi_loc = float(np.mean(channel_samples_log))
#         else:
#             roi_loc = float(np.median(channel_samples_log))
        
#         # Calculate standard deviation in log space
#         roi_scale = float(np.std(channel_samples_log)) * sigma_multiplier
        
#         # Store in dictionary (loc and scale in log space)
#         roi_dict[channel] = (roi_loc, roi_scale)
    
#     return roi_dict

# # Example usage:
# # roi_dict = extract_roi_from_trained_model(mmm)
# # print(roi_dict)

def get_variable_names(
    config_file=None,
    df=None,
    spend_pattern: str = 'spend_',
    impression_pattern: str = 'impressions_',
    organic_pattern: str = 'organic'):

    """
    Get the variable names from a config file or a dataframe
    Args:
        config_file: The path to the config file
        df: The dataframe to get the variable names from
    Returns:
        spend_cols, impression_cols, channels
    """
    if config_file is not None:
        import yaml
        with open(config_file, 'r') as f:
            config_yaml = yaml.safe_load(f)
        variable_names = config_yaml.get('variable_names', {})
        spend_cols = variable_names.get('spend_cols', [])
        impression_cols = variable_names.get('impression_cols', [])
        channels = variable_names.get('channels', [])        
    else:
        spend_cols = [c for c in df.columns if spend_pattern in c and organic_pattern not in c]
        # add spend_coop to impression_cols
        impression_cols = [c for c in df.columns if impression_pattern in c and organic_pattern not in c]
        channels = [c.replace(spend_pattern, '') for c in spend_cols]

        print(f"media channels found: {len(spend_cols)}")
        print(f"\nspend channels:")
        for col in sorted(spend_cols):
            print(f"  - {col}")

        print(f"\nimpression channels found: {len(impression_cols)}")
        print(f"\nimpression channels:")
        for col in sorted(impression_cols):
            print(f"  - {col}")

        print(f"\nmedia channels found: {len(channels)}")
        print(f"media channels:")
        for col in channels:
            print(f"  - {col}")
    
    return spend_cols, impression_cols, channels


# def export_parameters_to_yaml(mmm_model, 
#                               model_parameters = None, 
#                               roi_dictionary = None, 
#                               use_posterior_mean=True, 
#                               sigma_multiplier= 0.5, #defaults to 1 which is pretty loose, the lower this number is, the more constraint it will be 
#                               assume_log_space=False,# Set to True if roi_m is already in log space
#                               BUCKET_NAME = None,
#                               PARAMETERS_PATH = None,
#                               model_fname_prev = None, # TODO: remove this if necessary
#                               output_yaml_file = None):
#     """
#     Export model parameters and ROI priors from a trained Meridian model to a YAML file.
    
#     This function extracts model specification and ROI posterior statistics from a trained model
#     and saves them to a YAML file for reuse as priors in future model training.
    
#     Args:
#         mmm_model: A trained Meridian model object to export parameters from.
#         model_parameters: Optional dict of model parameters. If None, extracted from mmm_model.
#         roi_dictionary: Optional dict of ROI priors in format {'channel': (loc, scale)}.
#                        If None, extracted from mmm_model posterior.
#         use_posterior_mean: If True, use posterior mean for ROI estimates. If False, use median.
#         sigma_multiplier: Multiplier for ROI standard deviation (default 0.5).
#                          Use < 1.0 to tighten priors, > 1.0 to loosen them.
#         assume_log_space: If True, assumes roi_m posteriors are already in log space.
#                          If False (default), transforms from original scale to log space.
#         output_yaml_file: Path/filename for the output YAML file.
#         BUCKET_NAME: The name of the GCS bucket
#         PARAMETERS_PATH: The path to the parameters in the GCS bucket
#         model_fname_prev: The name of the previous model
    
#     Returns:
#         None. Writes YAML file to specified path with structure:
#         {
#             'model_spec': {...model parameters...},
#             'priors': {'roi_m': {...ROI dictionaries...}}
#         }
    
#     Example:
#         export_parameters_to_yaml(previous_mmm, 
#                                  sigma_multiplier=0.5,
#                                  output_yaml_file='model_priors.yaml')
#     """
#     # Prepare YAML data for export
#     if not model_parameters:
#         model_parameters = get_model_spec_attrs(mmm_model)
#     if not roi_dictionary:
#         roi_dictionary = extract_roi_from_trained_model(
#             mmm_model, 
#             use_posterior_mean=use_posterior_mean, 
#             sigma_multiplier=sigma_multiplier,
#             assume_log_space=assume_log_space 
#         )

#     yaml_data = {}

#     # Add model_parameters except for 'holdout_id' to 'model_spec'
#     model_spec_section = {}
#     for k, v in model_parameters.items():
#         if k == 'holdout_id':
#             continue
#         # Convert numpy arrays to lists for YAML serialization
#         if isinstance(v, np.ndarray):
#             model_spec_section[k] = v.tolist()
#         # Convert other tuples to lists for YAML
#         elif isinstance(v, tuple):
#             model_spec_section[k] = list(v)
#         elif isinstance(v, (np.generic,)):
#             model_spec_section[k] = v.item()
#         else:
#             model_spec_section[k] = v

#     yaml_data['model_spec'] = model_spec_section

#     # Ensure roi_m_dictionary_from_prev_model is properly serialized (already sanitized earlier)
#     yaml_data['priors'] = {'roi_m': copy.deepcopy(roi_dictionary)}

#     # Write to YAML file
#     if not output_yaml_file:
#         output_yaml_file = model_fname_prev.replace('.pkl', '_parameters.yaml')
#     with open(output_yaml_file, 'w') as f:
#         client = storage.Client()
#         bucket = client.bucket(BUCKET_NAME)
#         blob = bucket.blob(f'{PARAMETERS_PATH}{output_yaml_file}')
#         blob.upload_from_string(yaml.dump(yaml_data, default_flow_style=False))
#     print(f"Model parameters and priors written to {BUCKET_NAME}/{PARAMETERS_PATH}{output_yaml_file}")

# example usage
# export_parameters_to_yaml(previous_mmm, output_yaml_file='AMAN_params_test.yaml')


# def get_model_spec_attrs(mmm_model):
#     """
#     Extract non-empty model specification attributes from a Meridian model object.
    
#     This function iterates through all attributes of a model's model_spec and returns
#     those that are non-empty and not the 'prior' attribute. It handles conversion of
#     numpy arrays and other special types for serialization.
    
#     Args:
#         mmm_model: A Meridian model object containing a model_spec attribute.
    
#     Returns:
#         dict: A dictionary containing non-empty model specification attributes,
#               with numpy arrays converted to lists and numpy scalars to Python types.
#               The 'prior' attribute is always excluded.
    
#     Example:
#         >>> non_empty_attrs = get_model_spec_attrs(previous_mmm)
#         >>> print(non_empty_attrs)
#         {'media_effects_dist': 'log_normal', 'max_lag': 8, ...}
#     """
#     non_empty_attrs = {}
#     for k, v in vars(mmm_model.model_spec).items():
#         if k == 'prior':
#             continue
#         if v is None:
#             continue
#         # Handle numpy arrays and other types
#         try:
#             if isinstance(v, np.ndarray):
#                 if v.size > 0:
#                     non_empty_attrs[k] = v
#             elif v != '':
#                 non_empty_attrs[k] = v
#         except (ValueError, TypeError):
#             # If comparison fails, include the value
#             non_empty_attrs[k] = v

#     return non_empty_attrs


# def get_variable_type(data, df):
#     """
#     Get the variable types and columns for the data
#     Args:
#         data: The data object (data_frame_input_data_builder.DataFrameInputData)
#         df: The dataframe (pandas.DataFrame new input data)
#     Returns:
#         variable_type: A dictionary containing the variable types and columns
#     """
#     variable_type = {}
#     variable_type['paid_media_channel'] = list(data.get_all_paid_channels())
#     # generate spend and impression columns
#     variable_type['spend_cols'] = [f'spend_{channel}' for channel in variable_type['paid_media_channel']]
#     variable_type['impression_cols'] = [f'impressions_{channel}' for channel in variable_type['paid_media_channel']]
#     if data.organic_media_channel is not None:
#         variable_type['organic_media_channel'] = sorted(set(data.organic_media_channel.values))
#         variable_type['organic_media_cols'] = [col for col in df.columns if any(org in col for org in variable_type['organic_media_channel'])]
#     else:
#         variable_type['organic_media_channel'] = None
#         variable_type['organic_media_cols'] = None
#     variable_type['non_media_treatments'] = list(data.non_media_treatments.values) if data.non_media_treatments is not None else None
#     variable_type['controls'] = list(data.controls.values) if data.controls is not None else None
#     variable_type['non_media_channel'] = list(data.non_media_channel.values) if data.non_media_channel is not None else None
    
#     return variable_type

# variable_type_dict = get_variable_type(data,df)
# variable_type_dict['paid_media_channel'], variable_type_dict['organic_media_channel'], variable_type_dict['organic_media_cols']


# new stuff - export parameters to yaml

def export_parameters_to_yaml(mmm_model: None,
                              data = None,
                              df: pd.DataFrame = None,
                              new_parameters_dict: dict = None,
                              config_file_loaded: dict = None,
                              use_posterior_mean: bool = True, 
                              sigma_multiplier: float = 0.5, #defaults to 1 which is pretty loose, the lower this number is, the more constraint it will be 
                              assume_log_space: bool = False,# Set to True if roi_m is already in log space
                              BUCKET_NAME: str = None,
                              PARAMETERS_PATH: str = None,
                              model_fname_prev: str= None,  # Optional argument: set to a string if prior model filename is needed, else leave as None
                              output_yaml_file: str = None):
    """
    Export model parameters and ROI priors from a trained Meridian model to a YAML file in Scene MX format.

    This function collects model specification, build/project parameters, variable type info, 
    and ROI prior/posterior statistics from a trained model. It compiles these into a single YAML file 
    for downstream use (e.g., as priors for future training runs, model transfer, or experiment tracking).

    Args:
        mmm_model:      Trained Meridian model to export parameters/priors from.
        data:           DataFrameInputData object used to help infer variable types (form Builder object).
        df:             DataFrame of input data (required for variable types).
        new_parameters_dict: If supplied, dictionary containing user-specified parameters. 
                            All keys missing here will be supplemented from the loaded config file or by extraction from the model.
        config_file_loaded:  (dict, optional) Loaded project config file to provide default project/build/variable parameters.
        use_posterior_mean:  If True, uses posterior mean for ROI priors; otherwise uses median.
        sigma_multiplier:    Scales the standard deviation of ROI priors for tighter or looser constraints.
        assume_log_space:    If True, assumes ROI mean/sd are already in log space (no transform applied).
        BUCKET_NAME:         Name of the Google Cloud Storage bucket.
        PARAMETERS_PATH:     Path (within bucket) where the parameters YAML should be saved.
        model_fname_prev:    (Optional) Name of previous model file; used to generate output YAML filename if not provided.
        output_yaml_file:    (Optional) Path/filename for YAML output. If not specified, auto-generated from previous model name.

    Returns:
        Dictionary with the following keys:
            'project_parameters': {...},    # Project-level config parameters
            'build_parameters': {...},      # Model build parameters
            'variable_types': {...},        # Channel grouping (media/organic/other)
            'priors': {...},                # ROI and other priors (from posterior or config)
            'model_spec': {...},            # Model specification parameters

    Example:
        export_parameters_to_yaml(
            previous_mmm, 
            sigma_multiplier=0.5,
            BUCKET_NAME="your-bucket",
            PARAMETERS_PATH="exports/",
            output_yaml_file="model_priors.yaml"
        )
    """
    if not new_parameters_dict:
        new_parameters_dict = {}
    
    # Prepare YAML data for export
    # if no new project parameters are provided, use the ones from the config file
    if 'project_parameters' not in new_parameters_dict.keys():
        print(f"No project parameters provided, using the ones from the config file: {config_file_loaded['project_parameters']}")
        new_parameters_dict['project_parameters'] = config_file_loaded['project_parameters']

    # if no new build parameters are provided, use the ones from the config file
    if 'build_parameters' not in new_parameters_dict.keys():
        print(f"No build parameters provided, using the ones from the config file: {config_file_loaded['build_parameters']}")
        new_parameters_dict['build_parameters'] = config_file_loaded['build_parameters']

    if 'variable_types' not in new_parameters_dict.keys():
        print(f"No variable types provided, using the ones from the data: {get_variable_type(data,df)}")
        new_parameters_dict['variable_types'] = get_variable_type(data,df)

    if 'priors' not in new_parameters_dict.keys():
        print(f"No priors provided, using the ones from the model with extract_roi_from_trained_model()")
        new_parameters_dict['priors'] = extract_roi_from_trained_model(
            mmm_model, 
            use_posterior_mean=use_posterior_mean, 
            sigma_multiplier=sigma_multiplier,
            assume_log_space=assume_log_space 
        )

    if 'model_spec' not in new_parameters_dict.keys():
        print(f"No model spec provided, using the ones from the model: {get_model_spec_params(mmm_model)}")
        new_parameters_dict['model_spec'] = get_model_spec_params(mmm_model)

    if 'sampling_parameters' not in new_parameters_dict.keys():
        print(f"No sampling parameters provided, using the ones from the config file: {config_file_loaded['sampling_parameters']}")
        new_parameters_dict['sampling_parameters'] = config_file_loaded['sampling_parameters']

    if 'optimization_parameters' not in new_parameters_dict.keys():
        print(f"No optimization parameters provided, using the ones from the config file: {config_file_loaded['optimization_parameters']}")
        new_parameters_dict['optimization_parameters'] = config_file_loaded['optimization_parameters']

    # Write to YAML file
    if not output_yaml_file:
        output_yaml_file = model_fname_prev.replace('.pkl', '_parameters.yaml')
    with open(output_yaml_file, 'w') as f:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f'{PARAMETERS_PATH}{output_yaml_file}')
        blob.upload_from_string(yaml.dump(new_parameters_dict, default_flow_style=False))
    print(f"Model parameters and priors written to {BUCKET_NAME}/{PARAMETERS_PATH}{output_yaml_file}")
    return new_parameters_dict

def get_model_spec_params(mmm):
    """
    Retrieve not None arguments to a dictionary except for 'holdout_id'
    Args:
        mmm: The trained Meridian model object
    Returns:
        model_spec_dict: A dictionary containing the model specifications
    """
    # Retrieve not None arguments to a dictionary except for 'holdout_id'
    model_spec_dict = {}
    for k, v in mmm.model_spec.__dict__.items():
        if k != 'holdout_id' and k != 'prior' and v is not None:
            model_spec_dict[k] = v
    return model_spec_dict

# model_spec_dict = get_model_spec_params(mmm)

def get_variable_type(data, df):
    """
    Get the variable types and columns for the data
    Args:
        data: The data object (data_frame_input_data_builder.DataFrameInputData)
        df: The dataframe (pandas.DataFrame new input data)
    Returns:
        variable_type: A dictionary containing the variable types and columns
    """
    variable_type = {}
    variable_type['paid_media_channel'] = list(data.get_all_paid_channels())
    # generate spend and impression columns
    variable_type['spend_cols'] = [f'spend_{channel}' for channel in variable_type['paid_media_channel']]
    variable_type['impression_cols'] = [f'impressions_{channel}' for channel in variable_type['paid_media_channel']]
    if data.organic_media_channel is not None:
        variable_type['organic_media_channel'] = sorted(set(data.organic_media_channel.values))
        variable_type['organic_media_cols'] = [col for col in df.columns if any(org in col for org in variable_type['organic_media_channel'])]
    else:
        variable_type['organic_media_channel'] = None
        variable_type['organic_media_cols'] = None
    # variable_type['non_media_treatments'] = list(data.non_media_treatments.values) if data.non_media_treatments is not None else None
    variable_type['controls'] = list(data.control_variable.values) if data.controls is not None else None
    variable_type['non_media_channel'] = list(data.non_media_channel.values) if data.non_media_channel is not None else None
    
    return variable_type


# If you have a trained model loaded, you can extract its ROI posterior statistics
# to use them as priors for a new model training

def extract_roi_from_trained_model(
    trained_mmm, 
    use_posterior_mean=True, 
    sigma_multiplier=1.0, 
    assume_log_space=False):
    """
    Extract ROI statistics from a trained model and convert to roi_m_dictionary format.
    
    IMPORTANT: Since LogNormal priors require loc and scale parameters in log space,
    this function transforms ROI values from original scale to log space.
    
    Args:
        trained_mmm: A trained Meridian model object
        use_posterior_mean: If True, use posterior mean as ROI estimate. If False, use median.
        sigma_multiplier: Multiplier for standard deviation (default 1.0). 
                          Use < 1.0 to tighten priors, > 1.0 to loosen them.
        assume_log_space: If True, assumes roi_m posterior samples are already in log space.
                          If False (default), assumes they're in original scale and transforms them.
    
    Returns:
        Dictionary in format: {'channel_name': (roi_loc, roi_scale)} where loc and scale
        are in log space (ready for LogNormal distribution).
    """
    # Get ROI posterior samples
    roi_samples = trained_mmm.inference_data.posterior['roi_m'].values
    
    # Get channel names
    channel_names = trained_mmm.input_data.media_channel.values
    
    # Calculate statistics
    roi_dict = {}
    
    for i, channel in enumerate(channel_names):
        # Get posterior samples for this channel
        channel_samples = roi_samples[:, :, i].flatten()  # Flatten across chains and draws
        
        # Transform to log space if samples are in original scale
        if not assume_log_space:
            # Ensure all values are positive before taking log
            channel_samples = np.maximum(channel_samples, 1e-6)
            channel_samples_log = np.log(channel_samples)
        else:
            channel_samples_log = channel_samples
        
        # Calculate mean and std in log space (for LogNormal loc and scale parameters)
        if use_posterior_mean:
            roi_loc = float(np.mean(channel_samples_log))
        else:
            roi_loc = float(np.median(channel_samples_log))
        
        # Calculate standard deviation in log space
        roi_scale = float(np.std(channel_samples_log)) * sigma_multiplier
        
        # Store in dictionary (loc and scale in log space)
        roi_dict[channel] = [roi_loc, roi_scale]
    
    return roi_dict


# TODO(david.hidasi@assemblyglobal.com): this may be redundant, remove this if necessary
def get_channel_names(
    config_file=None, # TODO(david.hidasi@assemblyglobal.com): thjis has to be a loaded config file instead of a file path
    df=None):

    if config_file is not None:
        import yaml
        with open(config_file, 'r') as f:
            config_yaml = yaml.safe_load(f)
        variable_names = config_yaml.get('variable_names', {})
        spend_cols = variable_names.get('spend_cols', [])
        impression_cols = variable_names.get('impression_cols', [])
        channels = variable_names.get('channels', [])        
    else:
        spend_cols = sorted([c for c in df.columns if c.startswith('spend_')])
        # add spend_coop to impression_cols
        impression_cols = sorted([c for c in df.columns if c.startswith('impressions_')])
        channels = sorted([c.replace('spend_', '') for c in spend_cols])

        print(f"media channels found: {len(spend_cols)}")
        print(f"\nspend channels:")
        for col in sorted(spend_cols):
            print(f"  - {col}")

        print(f"\nimpression channels found: {len(impression_cols)}")
        print(f"\nimpression channels:")
        for col in sorted(impression_cols):
            print(f"  - {col}")

        print(f"\nmedia channels found: {len(channels)}")
        print(f"media channels:")
        for col in channels:
            print(f"  - {col}")
    return spend_cols, impression_cols, channels


def read_and_concat_parquet(out_folder):
    """
    Read and concatenate all parquet files in a folder
    Args:
        out_folder: The folder containing the parquet files
    Returns:
        df_all: A dataframe containing the concatenated data
    """
    opt_files = [f for f in os.listdir(out_folder) if f.endswith(".parquet")]
    df_all = pd.concat([pd.read_parquet(os.path.join(out_folder, f)) for f in opt_files], ignore_index=True)
    return df_all



def recreate_input_data(mmm_input_data):
    """
    This function recreates the input data from the mmm_input_data object.
    Args:
        mmm_input_data: The input data object from the mmm model.
    Returns:
        df_wide: A pandas DataFrame containing the input data.
    """
    # Each xarray DataArray in input_data can be converted to a DataFrame (coordinates become index/columns)
    # KPI: (geo, time) -> long table
    df_kpi = mmm_input_data.kpi.to_dataframe().reset_index()

    # Media (impressions): (geo, media_time, media_channel) — long table; wide cols get "impressions_" prefix
    df_media = mmm_input_data.media.to_dataframe().reset_index()
    media_wide = df_media.pivot_table(
        index=["geo", "media_time"], columns="media_channel", values="media"
    ).reset_index()
    media_wide = media_wide.rename(columns={c: f"impressions_{c}" for c in media_wide.columns if c not in ("geo", "media_time")})

    # Media spend: (geo, time, media_channel) — long table; wide cols get "spend_" prefix
    df_media_spend = mmm_input_data.media_spend.to_dataframe().reset_index()
    spend_wide = df_media_spend.pivot_table(
        index=["geo", "time"], columns="media_channel", values="media_spend"
    ).reset_index()
    spend_wide = spend_wide.rename(columns={c: f"spend_{c}" for c in spend_wide.columns if c not in ("geo", "time")})

    # Organic media: (geo, media_time, organic_media_channel)
    if mmm_input_data.organic_media is not None:
        df_organic_media = mmm_input_data.organic_media.to_dataframe().reset_index()
        organic_wide = df_organic_media.pivot_table(
            index=["geo", "media_time"], columns="organic_media_channel",
            values="organic_media"
        ).reset_index().rename(columns={"media_time": "time"})
    else:
        organic_wide = None

    # Non-media treatments: (geo, time, non_media_channel)
    if mmm_input_data.non_media_treatments is not None:
        df_non_media = mmm_input_data.non_media_treatments.to_dataframe().reset_index()
        non_media_wide = df_non_media.pivot_table(
            index=["geo", "time"], columns="non_media_channel",
            values="non_media_treatments"
        ).reset_index()
    else:
        non_media_wide = None

    # Controls: (geo, time, control_variable)
    if mmm_input_data.controls is not None:
        df_controls = mmm_input_data.controls.to_dataframe().reset_index()
        controls_wide = df_controls.pivot_table(
            index=["geo", "time"], columns="control_variable",
            values="controls"
        ).reset_index()
    else:
        controls_wide = None

    if mmm_input_data.population is not None:
        df_population = mmm_input_data.population.to_dataframe().reset_index()
        # population_wide = df_population.pivot_table(
        #     index=["geo", "media_time"], columns="population_variable",
        #     values="population"
        # ).reset_index().rename(columns={"media_time": "time"})
    else:
        df_population = None

    # One wide DataFrame — KPI + impressions + spend + organic_media + non_media + controls
    df_wide = df_kpi.merge(spend_wide, on=["geo", "time"], how="left")
    df_wide = df_wide.merge(
        media_wide.rename(columns={"media_time": "time"}),
        on=["geo", "time"],
        how="left",
    )
    if organic_wide is not None:
        df_wide = df_wide.merge(organic_wide, on=["geo", "time"], how="left")
    if non_media_wide is not None:
        df_wide = df_wide.merge(non_media_wide, on=["geo", "time"], how="left")
    if controls_wide is not None:
        df_wide = df_wide.merge(controls_wide, on=["geo", "time"], how="left")
    if df_wide is not None:
        df_wide = df_wide.merge(df_population, on=["geo"], how="left")

    return df_wide

def build_opt_constraints(mmm, custom_constraints_patterns, default_value=0.3):
    """
    Build a dictionary of spend constraints for each channel, assign default_value to channels not in custom_constraints_patterns
    Args:
        mmm: The trained Meridian model object
        custom_constraints_patterns: A dictionary of custom constraints patterns. e.g. {'metasearch': 0.2, 'brand': 0.2, 'pmax': 0.2}
        default_value: The default value to assign to channels not in custom_constraints_patterns
    Returns:
        spend_constraint_lower: A dictionary of spend constraints lower
        spend_constraint_upper: A dictionary of spend constraints upper
    """
    # The error means build_channel_args expects a keyword for *every* paid channel, even if not constrained!
    # So: you must set ALL expected channel keywords, not just the ones in your config, else you get ValueError

    # Get expected channel names
    expected_channels = list(mmm.input_data.media_channel.values)

    spend_constraint = {}
    for channel in expected_channels:
        channel_str = str(channel)
        constraint_val = default_value
        if custom_constraints_patterns:
            for pattern, constraint_value in custom_constraints_patterns.items():
                if pattern in channel_str:
                    constraint_val = constraint_value
                    break
        spend_constraint[channel_str] = constraint_val
    
    build_channel_args = mmm.input_data.get_paid_channels_argument_builder()
    spend_constraint_lower = build_channel_args(**spend_constraint)
    spend_constraint_upper = build_channel_args(**spend_constraint)
    return spend_constraint_lower, spend_constraint_upper