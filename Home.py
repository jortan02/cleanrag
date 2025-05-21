import streamlit as st
import asyncio
import torch
from utils.api_config import is_app_configured
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
from utils.api_config import (
    update_api_key,
    get_api_key,
    update_model_settings,
    get_model_settings,
    validate_openai_key
)
from utils.document_utils import (
    extract_text_from_file,
    process_document,
    get_chunk_statistics,
    create_chunk_dataframe,
    create_index_from_documents,
    VectorStoreType,
)
from llama_index.core import Document
from config import SUPPORTED_FILE_TYPES, SUPPORTED_QA_FILE_TYPES

# Callback functions
def handle_api_key_change():
    """Handle API key changes and validate"""
    api_key = st.session_state.openai_key
    if api_key:
        is_valid = validate_openai_key(api_key)
        st.session_state.api_key_valid = is_valid
        if is_valid:
            update_api_key("openai", api_key)
    else:
        st.session_state.api_key_valid = False

def handle_model_change():
    """Handle model selection changes"""
    if st.session_state.get("api_key_valid", False):
        update_model_settings(
            embedding_model=st.session_state.embedding_model,
            llm_model=st.session_state.llm_model,
            llm_provider="openai"  # Since we only use OpenAI now
        )

def handle_file_upload():
    """Handle file uploads"""
    uploaded_files = st.session_state.file_uploader
    if uploaded_files:
        for file in uploaded_files:
            store_uploaded_file(file)

def handle_splitter_change():
    """Handle splitter type changes"""
    # Get current options
    current_options = get_processing_options()
    # Update with new splitter type but keep other options
    update_processing_options(
        current_options[0],  # param1
        current_options[1],  # param2
        st.session_state.splitter_type,  # new splitter type
        current_options[3]  # vector_store_type
    )

def handle_processing_options_change():
    """Handle processing options changes"""
    if st.session_state.splitter_type == "sentence":
        update_processing_options(
            st.session_state.chunk_size,
            st.session_state.chunk_overlap,
            st.session_state.splitter_type,
            st.session_state.vector_store_type
        )
    else:  # semantic
        update_processing_options(
            st.session_state.buffer_size,
            st.session_state.breakpoint_threshold,
            st.session_state.splitter_type,
            st.session_state.vector_store_type
        )

# Initialize session state
initialize_session_state()
torch.classes.__path__ = []

# Setup asyncio event loop
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Page config
st.set_page_config(
    page_title="CleanRAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Navigation
st.sidebar.title("Navigation")

st.sidebar.markdown('''
- [Home](#home)
- [Setup](#setup)
- [Upload](#upload)
- [Chat](#chat)
- [Evaluate](#evaluate)
''', unsafe_allow_html=True)

# Home Section
st.header('Welcome to CleanRAG 🧹', anchor="home")
st.markdown("""
### Your RAG Pipeline Optimization Tool

CleanRAG helps you analyze, optimize, and debug your Retrieval-Augmented Generation (RAG) pipeline.

#### Quick Start:
1. **Setup**: Configure your OpenAI API key and model settings
2. **Upload**: Add your documents and process them
3. **Chat**: Interact with your documents
4. **Evaluate**: Analyze your RAG pipeline's performance

#### Current Status:
""")

# Show current status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        "API Key",
        "✅ Configured" if st.session_state.get("api_key_valid", False) else "❌ Not Configured", border=True
    )
with col2:
    docs = get_all_processed_documents()
    st.metric(
        "Documents",
        f"{len(docs)} Processed" if docs else "No Documents", border=True
    )
with col3:
    st.metric(
        "Index",
        "✅ Ready" if "index" in st.session_state else "❌ Not Created", border=True
    )

# Setup Section
st.header("Setup ⚙️", anchor="setup")

# API Keys Section
st.header("API Keys")

# OpenAI API Key with callback
api_key = st.text_input(
    "OpenAI API Key",
    value=get_api_key("openai"),
    type="password",
    help="Required for GPT models and OpenAI embeddings",
    key="openai_key",
    on_change=handle_api_key_change
)

# Show validation status
if "api_key_valid" in st.session_state:
    if st.session_state.api_key_valid:
        st.success("✅ API key is valid")
    else:
        st.error("❌ Invalid API key. Please check and try again.")
else:
    st.warning("⚠️ Please enter your OpenAI API key to continue")

# Only show model settings if API key is valid
if st.session_state.get("api_key_valid", False):
    # Model Settings Section
    st.header("Model Settings")
    
    # Get current settings
    current_settings = get_model_settings()
    
    # LLM Model Selection with callback
    st.selectbox(
        "OpenAI Model",
        options=[
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ],
        index=0 if current_settings["llm_model"] == "gpt-3.5-turbo" else 1,
        help="Select the OpenAI model to use",
        key="llm_model",
        on_change=handle_model_change
    )
    
    # Embedding Model Selection with callback
    st.selectbox(
        "Embedding Model",
        options=[
            "text-embedding-3-small",
            "text-embedding-3-large"
        ],
        index=0 if current_settings["embedding_model"] == "text-embedding-3-small" else 1,
        help="Select the OpenAI embedding model to use",
        key="embedding_model",
        on_change=handle_model_change
    )

# Upload Section
st.header("Upload & Process 📤", anchor="upload")

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
        st.write("")
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

# Chat Section
st.header("Chat 💬", anchor="chat")
# Chat interface will go here

# Evaluate Section
st.header("Evaluate 📊", anchor="evaluate")
# Evaluation interface will go here
