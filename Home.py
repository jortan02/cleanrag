import os
import torch
import streamlit as st
import pandas as pd
import numpy as np
from config import APP_NAME, APP_ICON, LOCAL_MODE, DEBUG, USE_GPU
from utils.gpu_utils import get_cuda_info, format_gpu_info

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide"
)

# Display mode information
mode_status = "Local Mode" if LOCAL_MODE else "Hosted Mode"
st.sidebar.markdown(f"**Running in:** {mode_status}")

# Create a sidebar
with st.sidebar:
    st.header("Settings")
    
    # Display GPU information if in local mode and GPU is enabled
    if LOCAL_MODE and USE_GPU:
        st.markdown("---")
        st.subheader("GPU Information")
        gpu_info = get_cuda_info()
        st.text(format_gpu_info(gpu_info))
        
        if DEBUG and gpu_info["cuda_available"] and gpu_info["nvidia_smi_output"]:
            with st.expander("Detailed GPU Info"):
                st.code(gpu_info["nvidia_smi_output"])
    
    # Debug information (only shown in debug mode)
    if DEBUG:
        st.markdown("---")
        st.markdown("### Debug Information")
        st.json({
            "local_mode": LOCAL_MODE,
            "debug_mode": DEBUG,
            "gpu_enabled": USE_GPU
        })

st.title("Welcome to the Home Page")
st.markdown("""
This is the main landing page of our application. Here you can:
- Navigate to different sections using the sidebar
- View key metrics and information
- Access the main features of the application
""")

# Example of a metric display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Active Users", value="1,234", delta="123")
with col2:
    st.metric(label="Total Sessions", value="5,678", delta="456")
with col3:
    st.metric(label="Success Rate", value="98%", delta="2%")

# Example of a chart
st.subheader("Sample Data Visualization")
chart_data = {
    "Category A": [1, 2, 3, 4, 5],
    "Category B": [2, 3, 4, 5, 6],
    "Category C": [3, 4, 5, 6, 7]
}
st.line_chart(chart_data)

# Add a footer
st.markdown("---")
st.markdown(f"Built with ❤️ using Streamlit | Running in {mode_status}") 