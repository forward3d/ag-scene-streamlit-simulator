from streamlit.testing.v1 import AppTest

def test_app_boots_without_errors():
    """Test that the Streamlit app loads without throwing exceptions."""
    # Initialize the AppTest framework pointing to your main app file
    at = AppTest.from_file("code/app.py")
    
    # Run the app
    at.run(timeout=15)
    
    # Assert that no uncaught exceptions were thrown during boot
    assert not at.exception