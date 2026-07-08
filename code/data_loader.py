"""Template-generic data loading layer.
Contains reusable GCS-first loading logic for any Streamlit datascience app.
App-specific loaders (e.g. Meridian model loading) belong in separate modules.
Local file fallback is only available in development (APP_ENV=development).
In production, GCS is the only data source — failures surface immediately to the user.
Local development setup:
  Place test data files manually in data/ (gitignored — never commit them).
  APP_ENV=development is set automatically via .taskfile.env when using `task run-app`.
"""
import io
import os
from pathlib import Path
import pandas as pd
import streamlit as st
import yaml
from google.cloud import storage

ROOT_DIR = Path(__file__).parent.parent

def _is_dev_environment() -> bool:
    """Returns True only when APP_ENV is explicitly set to 'development'."""
    return os.environ.get("APP_ENV", "").lower() == "development"

def load_config(config_path=ROOT_DIR / "config.yaml"):
    """Reads the YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def load_client_dataset(organisation_slug: str, brand_slug: str) -> pd.DataFrame:
    """
    Loads the client dataset.
    Tries GCS first. In development, falls back to a local file if GCS fails.
    In production, GCS failure raises immediately with a user-visible error.
    """
    config = load_config()
    client_conf = config["clients"][organisation_slug][brand_slug]

    try:
        storage_client = storage.Client(project=client_conf["gcp_project_id"])
        bucket = storage_client.get_bucket(client_conf["bucket_name"])
        df = pd.read_csv(
            io.BytesIO(bucket.blob(blob_name=client_conf["gcs_file_key"]).download_as_string()),
            encoding="UTF-8",
            sep=",",
        )
        st.success("Dataset loaded from GCS.")
        return df

    except Exception as e:
        if _is_dev_environment():
            st.warning(f"GCS load failed: {e}. Trying local fallback (development only)...")
            local_path = ROOT_DIR / client_conf["local_data_path"]
            if local_path.exists():
                return pd.read_csv(local_path)
            raise FileNotFoundError(
                f"No dataset found in GCS or local path: {local_path}"
            ) from e
        else:
            st.error(
                f"Failed to load dataset from GCS: {e}. "
                "GCS is the only data source in production. "
                "Check your GCP credentials and bucket configuration."
            )
            raise