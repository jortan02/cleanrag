import streamlit as st
import pandas as pd
from pathlib import Path
import time
from utils.document_utils import (
    extract_text_from_file,
    process_document,
    get_chunk_statistics,
    create_chunk_dataframe,
)
from utils.data_utils import (
    initialize_session_state,
    store_processed_document,
    get_all_processed_documents,
    store_qa_data,
    get_qa_data,
    update_processing_options,
    get_processing_options,
    clear_processed_documents,
    clear_qa_data,
)

initialize_session_state()

st.set_page_config(
    page_title="CleanRAG - Upload & Ingest", page_icon="📤", layout="wide"
)

st.title("Upload & Ingest Documents 📤")

# File upload section
st.header("Upload Documents")
uploaded_files = st.file_uploader(
    "Choose your documents", type=["txt", "pdf", "docx"], accept_multiple_files=True
)

# Processing options
st.header("Processing Options")
chunk_size = st.slider("Chunk Size", 100, 2000, get_processing_options()[0], step=100)
chunk_overlap = st.slider("Chunk Overlap", 0, 200, get_processing_options()[1], step=10)

# Update processing options when sliders change
update_processing_options(chunk_size, chunk_overlap)

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

                # Store results in session state
                document_data = {
                    "text": text,
                    "chunks": chunks,
                    "stats": stats,
                    "chunk_df": chunk_df,
                }
                store_processed_document(file.name, document_data)

                progress_bar.progress((i + 1) / len(uploaded_files))

            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

        st.success("Documents processed successfully!")
    else:
        st.warning("Please upload at least one document.")

# Display processed documents
processed_docs = get_all_processed_documents()

if processed_docs:
    for i, (filename, doc_data) in enumerate(processed_docs.items()):
        is_last = i == len(processed_docs) - 1
        with st.expander(f"📄 {filename}", expanded=is_last):
            # Add delete button for individual document
            if st.button(
                f"🗑️ Delete {filename}", key=f"delete_{filename}", type="secondary"
            ):
                processed_docs.pop(filename)
                st.rerun()

            st.markdown("### Document Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", doc_data["stats"]["total_chunks"])
            with col2:
                st.metric(
                    "Avg Chunk Size", f"{doc_data['stats']['avg_chunk_size']:.1f} words"
                )
            with col3:
                st.metric("Total Words", doc_data["stats"]["total_words"])

            st.markdown("### Chunk Details")
            st.dataframe(doc_data["chunk_df"], use_container_width=True)
    if st.button("🗑️ Clear All Documents", type="secondary"):
        clear_processed_documents()
        st.rerun()

# QA Test Set Section
st.header("QA Test Set")
st.markdown(
    """
Upload a QA test set to evaluate your RAG pipeline's performance. The test set should be in CSV format
with columns for questions and expected answers.
"""
)

qa_file = st.file_uploader("Upload QA Test Set", type=["csv"])

if qa_file:
    try:
        qa_data = pd.read_csv(qa_file)
        store_qa_data(qa_data)
        st.success("QA test set loaded successfully!")
    except Exception as e:
        st.error(f"Error loading QA test set: {str(e)}")

# Display all processed data
qa_data = get_qa_data()

# Display QA data
if qa_data is not None:
    st.dataframe(qa_data.head())
    if st.button("🗑️ Clear QA Test Set", type="secondary"):
        clear_qa_data()
        st.rerun()
