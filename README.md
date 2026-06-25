# **Streamlit Meridian MMM Dashboard Testing Template**

This project is a Streamlit-based dashboard for visualizing and optimizing Marketing Mix Models (MMM) using the Meridian library. It allows users to load datasets, visualize historical performance, and run budget optimization scenarios across different geographies.

## **📂 Project Structure**

The application is structured to support both local Windows execution and containerized Docker execution.

    .
    ├── config.yaml                     # Client abstraction layer configuration
    ├── code/
    │   ├── app.py                      # Base Streamlit application script
    │   ├── main_app.py                 # Advanced Streamlit app (bounds, response curves, forecasting)
    │   ├── data_loader.py              # Logic for GCS authentication and data fetching
    │   ├── session_manager.py          # Session state and URL parameter handling
    │   ├── test_loader.py              # Local testing script
    │   ├── scene_mx/                   # Custom local modules for advanced MMM modeling
    │   ├── style/
    │   │   └── style.css               # Custom styling
    │   └── images/                     # Logos and assets
    ├── data/                           # Local data storage (Excluded via .gitignore/.dockerignore)
    ├── Dockerfile                      # Production-ready Docker build configuration
    ├── .dockerignore                   # Security configuration to exclude sensitive data
    ├── uv.lock                         # Dependency lock file
    └── pyproject.toml                  # Project configuration


## **🚀 How to Run**

### **Prerequisites**

1. **Local Setup and Execution:** To run the application locally, you must first sync your project dependencies and authenticate with Google Cloud to ensure the script has the necessary permissions to access your GCP resources.

    **_Make sure you have `uv` installed, then run the following command to sync your environment and install necessary packages_**    

         uv sync

2. **GCloud CLI:** Ensure you have the Google Cloud SDK installed and authenticated.  
   gcloud auth login

   **_Make sure you login using approved credentials that have access to the GCS buckets defined in the scripts._**
   
    * gcloud auth application-default login --project=ag-stage-app-staging
    * set GOOGLE_APPLICATION_CREDENTIALS=%APPDATA%\gcloud\application_default_credentials.json
    * python -m streamlit run Test_Streamlit.py
        * **_Note_:**  http://localhost:8501/?org=assembly&brand=test_brand to see the data and model being loaded - otherwise its **Result Unknown**

3. **Changing Data Sources:** To add or update a client's data source, modify the config.yaml structure:

    ```
    clients:
        assembly:
            test_brand:
                gcp_project_id: "ag-stage-app-staging"
                bucket_name: "dundermifflin-dundermifflin-data-5rv8"
                gcs_file_key: "scene/dataset/generic/geo_all_channels.csv"
                gcs_model_key: "scene/models/generic/meridian_demo_speed-up-optimization.pkl"
                local_data_path: "data/geo_all_channels.csv"
                local_model_path: "data/meridian_demo_speed-up-optimization.pkl"
   ```

### **Other Option: Running Locally on Docker**

If your deployment environment requires Docker, or if you prefer to run the application in an isolated container, follow these steps (assuming a standard Dockerfile is present in your repository):

1. **Build the Docker image:** 
     ```
    docker build -t test-streamlit-app .
     ```

2. **Run the Docker container:** Ensure you map the default Streamlit port (8501) and mount your local GCP credentials into the container so the app can authenticate:
    
    ```
    docker run -p 8501:8501 \
      -v %APPDATA%\gcloud\application_default_credentials.json:/tmp/keys/creds.json:ro \
      -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/creds.json \
      test-streamlit-app
    ```

