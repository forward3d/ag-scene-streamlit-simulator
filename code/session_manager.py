import streamlit as st

def init_session():
    """Initializes standard session state variables and reads URL parameters."""
    
    # Initialize empty states for our data and model
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'smmm_model' not in st.session_state:
        st.session_state.smmm_model = None
        
    # Read the organization and brand from the URL parameters
    # Example URL: http://localhost:8501/?organisation_slug=assembly&brand_slug=test_brand
    st.session_state.identity_organisation_slug = st.query_params.get("organisation_slug", "unknown")
    st.session_state.identity_brand_slug = st.query_params.get("brand_slug", "unknown")