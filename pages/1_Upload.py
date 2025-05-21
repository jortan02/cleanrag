import streamlit as st
import pandas as pd
from pathlib import Path
import time
from utils.document_utils import (
    extract_text_from_file,
    process_document,
    get_chunk_statistics,
    create_chunk_dataframe,
    create_index_from_documents,
    VectorStoreType,
    get_file_size_mb,
    format_size_mb,
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
    get_uploaded_files,
    store_uploaded_file,
    clear_uploaded_files,
)
from llama_index.core import Document
from config import (
    SUPPORTED_FILE_TYPES,
    SUPPORTED_QA_FILE_TYPES,
)

# Initialize session state
initialize_session_state()

def handle_file_upload():
    """Handle newly uploaded files."""
    # Get the uploaded files directly from the widget's return value
    uploaded_files = st.session_state.file_uploader
    stored_files = get_uploaded_files()
    
    # Remove files that are no longer in the uploader
    current_filenames = {f.name for f in uploaded_files} if uploaded_files else set()
    for filename in list(stored_files.keys()):
        if filename not in current_filenames:
            stored_files.pop(filename)
    
    # Add new files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in stored_files:
                store_uploaded_file(file)

def handle_splitter_change():
    """Handle splitter type change."""
    if get_all_processed_documents():
        st.warning(
            "⚠️ Changing processing settings will clear all previously processed documents. "
            "Please reprocess your documents with the new settings."
        )
        clear_processed_documents()

def handle_processing_options_change():
    """Handle processing options change."""
    if get_all_processed_documents():
        st.warning(
            "⚠️ Changing processing settings will clear all previously processed documents. "
            "Please reprocess your documents with the new settings."
        )
        clear_processed_documents()
        update_processing_options(*get_current_options())

def get_current_options():
    """Get current processing options based on splitter type."""
    splitter_type = st.session_state.splitter_type
    if splitter_type == "sentence":
        return (
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
            splitter_type,
            st.session_state.vector_store_type
        )
    else:  # semantic
        return (
            st.session_state.buffer_size,
            st.session_state.breakpoint_threshold,
            splitter_type,
            st.session_state.vector_store_type
        )

st.set_page_config(
    page_title="CleanRAG - Upload & Ingest", page_icon="📤", layout="wide"
)

st.title("Upload & Ingest Documents 📤")

# File upload section
st.header("Upload Documents")

# File uploader with callback
st.file_uploader(
    "Choose your documents",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
    key="file_uploader",
    on_change=handle_file_upload
)

# Display stored files count and delete all button
stored_files = get_uploaded_files()
if stored_files:
    col1, col2 = st.columns([4, 1], vertical_alignment="center")
    with col1:
        st.markdown(f'<span style="font-size: 1.5em">📚</span> {len(stored_files)} Document{"s" if len(stored_files) > 1 else ""} Uploaded', unsafe_allow_html=True)
    with col2:
        st.write("")  # Add some vertical spacing
        if st.button("🗑️ Clear All Files", type="secondary", use_container_width=True):
            clear_uploaded_files()
            st.rerun()

# Processing options
st.header("Processing Options")

# Splitter type selection with callback
splitter_type = st.selectbox(
    "Splitter Type",
    options=["sentence", "semantic"],
    format_func=lambda x: "Sentence-based" if x == "sentence" else "Semantic",
    help="Choose how to split documents. Sentence-based splits at sentence boundaries, while semantic splitting preserves document structure.",
    key="splitter_type",
    on_change=handle_splitter_change
)

# Show different controls based on splitter type
if splitter_type == "sentence":
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=get_processing_options()[0],
            step=100,
            help="Target size for each chunk in tokens",
            key="chunk_size",
            on_change=handle_processing_options_change
        )
    with col2:
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=get_processing_options()[1],
            step=10,
            help="Number of tokens to overlap between chunks",
            key="chunk_overlap",
            on_change=handle_processing_options_change
        )
else:  # semantic
    col1, col2 = st.columns(2)
    with col1:
        buffer_size = st.slider(
            "Buffer Size",
            min_value=1,
            max_value=5,
            value=get_processing_options()[0],
            step=1,
            help="Number of sentences to buffer when splitting. Higher values may result in more natural splits.",
            key="buffer_size",
            on_change=handle_processing_options_change
        )
    with col2:
        breakpoint_threshold = st.slider(
            "Breakpoint Threshold",
            min_value=50,
            max_value=99,
            value=get_processing_options()[1],
            step=1,
            help="Percentile threshold for determining breakpoints. Higher values result in fewer splits.",
            key="breakpoint_threshold",
            on_change=handle_processing_options_change
        )

# Vector store options with callback
vector_store_type = st.selectbox(
    "Vector Store Type",
    options=["simple", "faiss"],
    format_func=lambda x: "Simple" if x == "simple" else "FAISS",
    help="Choose the vector store type. FAISS provides better performance for larger datasets.",
    key="vector_store_type",
    on_change=handle_processing_options_change
)

# Process button
if st.button("Process Documents", type="primary"):
    if stored_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, (filename, file_data) in enumerate(stored_files.items()):
            status_text.text(f"Processing {filename}...")

            try:
                # Create a BytesIO object from the stored content
                from io import BytesIO
                file = BytesIO(file_data["content"])
                file.name = filename
                file.type = file_data["type"]

                # Extract text from file
                text = extract_text_from_file(file)

                # Create LlamaIndex document
                doc = Document(text=text, metadata={"filename": filename})

                # Process into chunks with splitter-specific parameters
                if st.session_state.splitter_type == "sentence":
                    kwargs = {
                        "chunk_size": st.session_state.chunk_size,
                        "chunk_overlap": st.session_state.chunk_overlap
                    }
                else:  # semantic
                    kwargs = {
                        "buffer_size": st.session_state.buffer_size,
                        "breakpoint_percentile_threshold": st.session_state.breakpoint_threshold
                    }
                
                chunks = process_document(
                    text, 
                    splitter_type=st.session_state.splitter_type,
                    **kwargs
                )

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
                    "document": doc,  # Store the LlamaIndex document
                }
                store_processed_document(filename, document_data)

                progress_bar.progress((i + 1) / len(stored_files))

            except Exception as e:
                st.error(f"Error processing {filename}: {str(e)}")

        # Create the index
        try:
            documents = [doc_data["document"] for doc_data in get_all_processed_documents().values()]
            if documents:
                index = create_index_from_documents(
                    documents,
                    vector_store_type=st.session_state.vector_store_type
                )
                # Store the index in session state
                st.session_state.index = index
                st.success("Documents processed and indexed successfully!")
        except Exception as e:
            st.error(f"Error creating index: {str(e)}")
    else:
        st.warning("Please upload at least one document.")

# Display processed documents
processed_docs = get_all_processed_documents()

if processed_docs:
    for i, (filename, doc_data) in enumerate(processed_docs.items()):
        with st.expander(f"📄 {filename}", expanded=False):
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
        if "index" in st.session_state:
            del st.session_state.index
        st.rerun()

# QA Test Set Section
st.header("QA Test Set")
st.markdown(
    """
Upload a QA test set to evaluate your RAG pipeline's performance. The test set should be in CSV format
with columns for questions and expected answers.
"""
)

qa_file = st.file_uploader("Upload QA Test Set", type=SUPPORTED_QA_FILE_TYPES)

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
