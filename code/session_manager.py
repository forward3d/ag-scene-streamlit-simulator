import streamlit as st

def init_session():
    """Initializes standard session state variables and reads URL parameters."""
    
    # Initialize empty states for our data and model
    if 'dataset' not in st.session_state:
        st.session_state.dataset = None
    if 'smmm_model' not in st.session_state:
        st.session_state.smmm_model = None
        
    # Read the organization and brand from the URL parameters
    # Example URL: http://localhost:8501/?org=assembly&brand=test_brand
    st.session_state.identity_org = st.query_params.get("org", "unknown")
    st.session_state.identity_brand = st.query_params.get("brand", "unknown")