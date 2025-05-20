import streamlit as st
import os
import torch

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


st.set_page_config(
    page_title="CleanRAG - Overview",
    page_icon="🏠",
    layout="wide"
)

st.title("Welcome to CleanRAG 🧹")
st.markdown("""
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
1. **Upload Documents**: Go to the Upload & Ingest page to add your documents
2. **Configure Settings**: Adjust your RAG parameters in the Settings page
3. **Run Diagnostics**: Use the Diagnostics page to evaluate your pipeline
4. **Test Queries**: Try out searches in the Search & Exploration page
5. **View Results**: Check the Results & Visualization page for insights

#### Key Features:
- **Document Processing**: Automatic chunking and embedding generation
- **Quality Metrics**: Comprehensive evaluation of retrieval quality
- **Interactive Debugging**: Real-time analysis of retrieval behavior
- **Performance Visualization**: Clear insights into your pipeline's effectiveness
- **Configuration Management**: Easy adjustment of RAG parameters

Get started by navigating to the Upload & Ingest page to begin processing your documents!
""")

# Add a call-to-action button
if st.button("🚀 Start with Document Upload", type="primary"):
    st.switch_page("pages/2_Upload_Ingest.py") 