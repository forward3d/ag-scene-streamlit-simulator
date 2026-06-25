import pandas as pd
import streamlit as st
import yaml
import time
import tempfile
import os
import io
import types
from pathlib import Path
from google.cloud import storage

# Calculate ROOT_DIR relative to this file (code/data_loader.py)
ROOT_DIR = Path(__file__).parent.parent

def load_config(config_path=ROOT_DIR / "config.yaml"):
    """Reads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

@st.cache_resource
def get_meridian_resource():
    """Imports and initializes the Meridian client."""
    modules = types.SimpleNamespace()
    
    from meridian.data import test_utils, load
    modules.test_utils = test_utils
    modules.load = load
    
    from meridian.model import model
    modules.model = model
    from meridian.model import spec
    modules.spec = spec
    from meridian.model import prior_distribution
    modules.prior_distribution = prior_distribution

    from meridian.analysis import optimizer, analyzer, visualizer, summarizer, formatter
    modules.optimizer = optimizer
    modules.analyzer = analyzer
    modules.visualizer = visualizer
    modules.summarizer = summarizer
    modules.formatter = formatter
    
    time.sleep(3) 
    return modules

def load_client_dataset(org, brand):
    """Handles the triple-fallback logic for loading datasets."""
    config = load_config()
    client_conf = config['clients'][org][brand]
    
    try:
        # First Fallback: Try GCS
        storage_client = storage.Client(project=client_conf['gcp_project_id'])
        bucket = storage_client.get_bucket(client_conf['bucket_name'])
        
        df = pd.read_csv(
            io.BytesIO(bucket.blob(blob_name=client_conf['gcs_file_key']).download_as_string()),
            encoding='UTF-8', sep=','
        )
        st.success("✅ Dataset loaded from GCS!")
        return df
    except Exception as e:
        # Second Fallback: Local file
        st.warning(f"⚠️ GCS Load failed: {e}. Trying local fallback...")
        local_path = ROOT_DIR / client_conf['local_data_path']
        if local_path.exists():
            return pd.read_csv(local_path)
        else:
            raise FileNotFoundError(f"🚨 No dataset found in GCS or Local path: {local_path}")

def load_client_model(org, brand):
    """Handles the triple-fallback logic for loading Meridian models."""
    config = load_config()
    client_conf = config['clients'][org][brand]
    meridian = get_meridian_resource()
    temp_model_path = None
    
    try:
        # First Fallback: Try GCS
        storage_client = storage.Client(project=client_conf['gcp_project_id'])
        bucket = storage_client.get_bucket(client_conf['bucket_name'])
        blob = bucket.blob(blob_name=client_conf['gcs_model_key'])
        pickle_bytes = blob.download_as_bytes()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            temp_model_path = temp_file.name
            temp_file.write(pickle_bytes)
            
        model = meridian.model.load_mmm(temp_model_path)
        st.success("✅ Model loaded from GCS!")
        return model
        
    except Exception as e:
        st.warning(f"⚠️ GCS Model Load failed: {e}. Trying local fallback...")
        local_path = ROOT_DIR / client_conf['local_model_path']
        if local_path.exists():
            return meridian.model.load_mmm(str(local_path))
        else:
            raise FileNotFoundError(f"🚨 No model found in GCS or Local path: {local_path}")
    finally:
        if temp_model_path and os.path.exists(temp_model_path):
            os.remove(temp_model_path)