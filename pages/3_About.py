import streamlit as st
from config import APP_NAME, APP_ICON, LOCAL_MODE, USE_GPU
from utils.gpu_utils import get_cuda_info, format_gpu_info

st.set_page_config(
    page_title=f"{APP_NAME} - About",
    page_icon=APP_ICON,
    layout="wide"
)

st.title("About")
st.markdown("""
This application demonstrates a well-structured Streamlit project with:
- Multiple pages
- Configuration management
- GPU support
- Data analysis capabilities
- Clean code organization
""")

# System Information
st.header("System Information")
st.markdown("""
### Application Settings
- **Mode**: {'Local' if LOCAL_MODE else 'Hosted'}
- **GPU Support**: {'Enabled' if USE_GPU else 'Disabled'}
""")

# Display GPU information if in local mode
if LOCAL_MODE and USE_GPU:
    st.subheader("GPU Information")
    gpu_info = get_cuda_info()
    st.text(format_gpu_info(gpu_info))

# Project Structure
st.header("Project Structure")
st.markdown("""
```
my_streamlit_app/
│
├── Home.py                # Main Streamlit app entry point
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .env                   # Environment variables
├── .env.example           # Example environment file
│
├── config.py              # Configuration management
│
├── utils/                 # Utility modules
│   └── gpu_utils.py
│
├── pages/                 # Additional Streamlit pages
│   ├── 2_Analysis.py
│   └── 3_About.py
```
""")

# Contact Information
st.header("Contact")
st.markdown("""
For questions or support, please contact:
- Email: support@example.com
- GitHub: [Project Repository](https://github.com/yourusername/your-repo)
""") 