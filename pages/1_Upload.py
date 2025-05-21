import streamlit as st
import pandas as pd
from pathlib import Path
import time
from utils.document_utils import (
    extract_text_from_file,
    process_document,
    get_chunk_statistics,
    create_chunk_dataframe
)
from config import LOCAL_MODE

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
    type=['txt', 'pdf', 'docx'],
    accept_multiple_files=True
)

# Processing options
st.header("Processing Options")
chunk_size = st.slider("Chunk Size", 100, 2000, 500, step=100)
chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, step=10)

# Process button
if st.button("Process Documents", type="primary"):
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            # Extract text from file
            try:
                text = extract_text_from_file(file)
                
                # Process into chunks
                chunks = process_document(text, chunk_size, chunk_overlap)
                
                # Get chunk statistics
                stats = get_chunk_statistics(chunks)
                
                # Create chunk dataframe
                chunk_df = create_chunk_dataframe(chunks)
                
                # Display results in expandable sections
                with st.expander(f"📄 {file.name} - Extracted Text & Chunks", expanded=True):
                    # Show statistics
                    st.markdown("### Document Statistics")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Chunks", stats["total_chunks"])
                    with col2:
                        st.metric("Avg Chunk Size", f"{stats['avg_chunk_size']:.1f} words")
                    with col3:
                        st.metric("Total Words", stats["total_words"])
                    
                    # Show chunk details
                    st.markdown("### Chunk Details")
                    st.dataframe(chunk_df, use_container_width=True)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        st.success("Documents processed successfully!")
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
