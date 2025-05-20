import streamlit as st
import pandas as pd
from pathlib import Path
import time

st.set_page_config(
    page_title="CleanRAG - Upload & Ingest",
    page_icon="📤",
    layout="wide"
)

st.title("Upload & Ingest Documents 📤")

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose your documents",
    type=['txt', 'pdf', 'docx', 'md'],
    accept_multiple_files=True
)

# Processing options
st.header("Processing Options")
col1, col2 = st.columns(2)

with col1:
    chunk_size = st.slider("Chunk Size", 100, 2000, 500, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, step=10)

with col2:
    embedding_model = st.selectbox(
        "Embedding Model",
        ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
    )
    processing_mode = st.radio(
        "Processing Mode",
        ["Local", "Cloud"],
        horizontal=True
    )

# Process button
if st.button("Process Documents", type="primary"):
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate processing
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            time.sleep(1)  # Simulate processing time
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        st.success("Documents processed successfully!")
        
        # Display processing results
        st.header("Processing Results")
        results_data = {
            "Document": [f.name for f in uploaded_files],
            "Chunks": [10, 15, 8],  # Example data
            "Status": ["Processed"] * len(uploaded_files)
        }
        st.dataframe(pd.DataFrame(results_data))
    else:
        st.warning("Please upload at least one document.")

# QA Test Set Section
st.header("QA Test Set")
st.markdown("""
Upload a QA test set to evaluate your RAG pipeline's performance. The test set should be in CSV format
with columns for questions and expected answers.
""")

qa_file = st.file_uploader("Upload QA Test Set", type=['csv'])

if qa_file:
    try:
        qa_data = pd.read_csv(qa_file)
        st.success("QA test set loaded successfully!")
        st.dataframe(qa_data.head())
    except Exception as e:
        st.error(f"Error loading QA test set: {str(e)}") 