# -*- coding: utf-8 -*-
"""
Created on Mon May 21 22:31:44 2026

@author: AshnerGeraldNovilla
"""

# -*- coding: utf-8 -*-
# gcloud auth application-default login --project=ag-stage-app-staging

import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import warnings
from PIL import Image
import base64
from pathlib import Path

# --- Import our new modules ---
from session_manager import init_session
from data_loader import load_client_dataset, load_client_model, get_meridian_resource

# --- Warnings Filter ---
warnings.filterwarnings("ignore")

# --- Global Variables ---
date_column = "time"
geo_column = "geo"
BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent
logo_img = Image.open(BASE_DIR / "images" / "assembly_icon.png")
logo_path = str(BASE_DIR / "images" / "logo.svg")

# --- UI Helper Functions ---
def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def selector_filter(file_path=None):
    data = file_path
    st.subheader("Modeling Selections and Configuration")    
    col1, col2, col3 = st.columns(3)

    with col1:
        geo_option = st.multiselect("Geo Selection", tuple(data[geo_column].unique()), max_selections=1, key="geo_option", default=data[geo_column].unique()[0])
        if len(data[geo_column].unique()) <= 1:
            geo_option = 'national_geo'

    with col2:
        date_start = st.date_input("Start Date", pd.to_datetime('2023-10-30'))

    with col3:
        date_end = st.date_input("End Date", pd.to_datetime(data[date_column]).max(), max_value=pd.to_datetime(data[date_column]).max())
        
    start_date_compare = pd.to_datetime(date_start)
    end_date_compare = pd.to_datetime(date_end)
    dt = data.copy()
    dt[date_column] = pd.to_datetime(dt[date_column])
    dt = data.loc[(data[geo_column].isin(geo_option)) & (dt[date_column] >= start_date_compare) & (dt[date_column] <= end_date_compare)]
    
    st.subheader("Dataset Preview")
    with st.expander("🔼🔽 Dataset Preview", expanded=False):
        st.dataframe(pd.concat([dt.head(), dt.tail()], ignore_index=True), use_container_width=True, hide_index=True)

    return dt

# --- Page Configuration and CSS ---
st.set_page_config(layout='wide', page_title="MMM Dashboard", page_icon=logo_img, initial_sidebar_state="expanded")

# [KEEP ALL YOUR CUSTOM CSS HERE - Omitted for brevity but leave your logo/sidebar styling exactly as it was]
st.markdown("""
    <style>
    div[data-baseweb="select"] { max-height: 40px; overflow: auto; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- App Logic ---
init_session()
org = st.session_state.identity_org
brand = st.session_state.identity_brand

st.sidebar.title("🔐 Proof of Concept")
st.sidebar.title("🚀 Get Started: Media Mix Model Using Meridian")

# Gatekeeper: Replaces `user_name == 'Ashner'`
if org == "unknown" or brand == "unknown":
    st.title("Welcome to the Scene MX Dashboard")
    st.warning("⚠️ No valid client configuration detected.")
    st.info("Please provide a valid organization and brand in the URL. Example: `/?org=assembly&brand=test_brand`")
    st.stop()

st.title(f"Scene MX Dashboard - {brand.replace('_', ' ').title()}")
st.subheader(f'Welcome, viewing data for {org.upper()}')

# Load Data and Model using the abstracted config logic
with st.spinner(f"Loading data for {org}..."):
    try:
        if st.session_state.dataset is None:
            st.session_state.dataset = load_client_dataset(org, brand)
            
        if st.session_state.smmm_model is None:
            st.session_state.smmm_model = load_client_model(org, brand)
            
    except Exception as e:
        st.error(str(e))
        st.stop()

dataset = st.session_state.dataset
smmm = st.session_state.smmm_model

if dataset is not None:
    dt = selector_filter(file_path=dataset)

run_clicked = st.button("🏃💨 Run Model Summary", type="primary")

if run_clicked:
    meridian = get_meridian_resource()
    media_summary = meridian.visualizer.MediaSummary(smmm)
    st.write(media_summary.summary_table())