import os
import tempfile
import streamlit as st
import types
import time
from google.cloud import storage
from data_loader import load_config, _is_dev_environment, ROOT_DIR

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
    from meridian import constants
    modules.constants = constants


    from meridian.analysis import optimizer, analyzer, visualizer, summarizer, formatter
    modules.optimizer = optimizer
    modules.analyzer = analyzer
    modules.visualizer = visualizer
    modules.summarizer = summarizer
    modules.formatter = formatter
    
      # Custom Local Scene MX Modules ---
    from scene_mx import get_optimized_df, get_summary_df, get_response_curves_df
    modules.get_optimized_df = get_optimized_df
    modules.get_summary_df = get_summary_df
    modules.get_response_curves_df = get_response_curves_df
    
    # Pausing execution briefly to ensure heavy Meridian resources are fully initialized in memory
    time.sleep(3) 
    return modules

def load_client_model(org, brand):
    """Handles loading Meridian models with GCS-first and dev-only local fallback."""
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
        if _is_dev_environment():
            st.warning(f"⚠️ GCS Model Load failed: {e}. Trying local fallback (dev only)...")
            local_path = ROOT_DIR / client_conf['local_model_path']
            if local_path.exists():
                return meridian.model.load_mmm(str(local_path))
            raise FileNotFoundError(f"🚨 No model found in GCS or Local path: {local_path}")
        else:
            st.error("Failed to load model from GCS. Check credentials and bucket config.")
            raise
    finally:
        if temp_model_path and os.path.exists(temp_model_path):
            os.remove(temp_model_path)