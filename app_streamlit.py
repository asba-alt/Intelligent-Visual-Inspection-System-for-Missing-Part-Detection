import streamlit as st
import os
import requests
from PIL import Image
import io

# Configuration
FASTAPI_URL = "http://localhost:8000"
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Intelligent Visual Inspection System",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-label {
        font-weight: bold;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔍 Intelligent Visual Inspection System</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check API health
try:
    health_response = requests.get(f"{FASTAPI_URL}/health", timeout=2)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data.get("model_loaded"):
            st.success("✅ System ready - Model loaded")
        else:
            st.warning("⚠️ Model not loaded. Please train a model first.")
    else:
        st.error("❌ API connection failed")
except requests.exceptions.RequestException:
    st.error("❌ Cannot connect to FastAPI backend. Make sure it's running on port 8000.")
    st.info("Run: `python app_fastapi.py` or `uvicorn app_fastapi:app --host 0.0.0.0 --port 8000`")

# File uploader
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader(
    "Select an assembly image to analyze",
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image to classify as complete or defect"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, width="stretch")
    
    # Predict button
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                # Prepare file for upload
                uploaded_file.seek(0)
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                
                # Send to FastAPI
                response = requests.post(f"{FASTAPI_URL}/predict", files=files, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Grad-CAM overlay
                    with col2:
                        st.subheader("🎯 Grad-CAM Overlay")
                        if result.get('overlay_url'):
                            overlay_path = result['overlay_url'].replace('/static/', 'static/')
                            if os.path.exists(overlay_path):
                                overlay_img = Image.open(overlay_path)
                                st.image(overlay_img, width="stretch")
                            else:
                                st.warning("Overlay not available")
                        else:
                            st.warning("Grad-CAM not available")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    # Create metrics
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric(
                            label="Predicted Class",
                            value=result['predicted_class'].replace('_', ' ').upper()
                        )
                    
                    with result_col2:
                        confidence_percent = result['confidence'] * 100
                        st.metric(
                            label="Confidence",
                            value=f"{confidence_percent:.2f}%"
                        )
                    
                    with result_col3:
                        status_emoji = "✅" if result['status'] == "PASS" else ("❌" if result['status'] == "FAIL" else "⚠️")
                        st.metric(
                            label="Status",
                            value=f"{status_emoji} {result['status']}"
                        )
                    
                    # Top 3 predictions
                    st.markdown("### 🏆 Top 3 Predictions")
                    top3_col1, top3_col2, top3_col3 = st.columns(3)
                    
                    for idx, (col, pred) in enumerate(zip([top3_col1, top3_col2, top3_col3], result['top3_predictions'])):
                        with col:
                            st.metric(
                                label=f"#{idx+1}",
                                value=pred['class'].replace('_', ' ').title(),
                                delta=f"{pred['confidence']*100:.1f}%"
                            )
                    
                    # Detailed info
                    with st.expander("📋 Detailed Information"):
                        st.json(result)
                    
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# Instructions
with st.sidebar:
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Upload** an assembly image
    2. Click **Analyze Image**
    3. View the **prediction results** and **Grad-CAM** visualization
    
    ### Classification (10 Classes)
    - **Complete**: All parts present, no defects
    - **Bent Lead**: Lead/pin is bent
    - **Cut Lead**: Lead/pin is cut
    - **Damaged Case**: Case/body damage
    - **Manipulated Front**: Front tampering
    - **Misplaced**: Component misaligned
    - **Scratch Head**: Scratch on head
    - **Scratch Neck**: Scratch on neck
    - **Thread Side**: Side thread defect
    - **Thread Top**: Top thread defect
    
    ### Grad-CAM
    Shows which areas of the image influenced the model's decision.
    
    ### System Requirements
    - FastAPI backend must be running
    - Trained model must be available
    """)
    
    st.markdown("---")
    st.caption("Intelligent Visual Inspection System v2.0 (Multi-Class)")
