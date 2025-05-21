import streamlit as st
import asyncio
import torch
from utils.api_config import is_app_configured
from utils.data_utils import initialize_session_state

initialize_session_state()
torch.classes.__path__ = []

st.set_page_config(page_title="CleanRAG - Overview", page_icon="🏠", layout="wide")

# if not is_app_configured():
#     st.navigation(["Home.py", "pages/0_Setup.py"])

st.title("Welcome to CleanRAG 🧹")
st.markdown(
    """
### Your RAG Pipeline Optimization Tool

CleanRAG helps you analyze, optimize, and debug your Retrieval-Augmented Generation (RAG) pipeline. 
Whether you're building a new RAG system or improving an existing one, CleanRAG provides the tools 
you need to ensure high-quality document retrieval and generation.

#### What You Can Do:
- 📤 Upload and process documents for indexing
- 🔍 Analyze retrieval effectiveness and quality
- 🎯 Test and evaluate your RAG pipeline
- ⚙️ Configure and optimize pipeline parameters
- 📊 Visualize performance metrics and results

#### Quick Start Guide:
1. **Configure API**: Go to the Setup page to add your OpenAI API key
2. **Upload Documents**: Go to the Upload & Ingest page to add your documents
3. **Run Diagnostics**: Use the Diagnostics page to evaluate your pipeline
4. **Test Queries**: Try out searches in the Search & Exploration page
5. **View Results**: Check the Results & Visualization page for insights

#### Key Features:
- **Document Processing**: Automatic chunking and embedding generation
- **Quality Metrics**: Comprehensive evaluation of retrieval quality
- **Interactive Debugging**: Real-time analysis of retrieval behavior
- **Performance Visualization**: Clear insights into your pipeline's effectiveness
- **Configuration Management**: Easy adjustment of RAG parameters
"""
)

# Add a call-to-action button
if not is_app_configured():
    st.warning(
        "⚠️ Please configure your OpenAI API key in the Setup page before proceeding."
    )

if st.button("⚙️ Go to Setup", type="primary"):
    st.switch_page("pages/0_Setup.py")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
