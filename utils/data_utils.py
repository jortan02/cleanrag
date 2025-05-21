from typing import List, Dict, Any, Optional
import streamlit as st
from pathlib import Path
import pandas as pd
from utils.document_utils import DocumentChunk
from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BREAKPOINT_THRESHOLD
)


def initialize_session_state():
    """Initialize all session state variables used across the application."""
    # Document processing state
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}

    # Uploaded files state
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}

    # QA test set state
    if "qa_data" not in st.session_state:
        st.session_state.qa_data = None

    # API and model settings
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "openai": "",
            "anthropic": "",
            "huggingface": ""
        }
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {
            "embedding_model": "text-embedding-3-small",
            "llm_model": "gpt-3.5-turbo",
            "llm_provider": "openai"
        }

    # Processing options state
    if "splitter_type" not in st.session_state:
        st.session_state.splitter_type = "sentence"
    if "sentence_options" not in st.session_state:
        st.session_state.sentence_options = {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP
        }
    if "semantic_options" not in st.session_state:
        st.session_state.semantic_options = {
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "breakpoint_threshold": DEFAULT_BREAKPOINT_THRESHOLD
        }
    if "vector_store_type" not in st.session_state:
        st.session_state.vector_store_type = "simple"


def save_processed_chunks(filename: str, chunks: List[DocumentChunk]):
    """Save processed chunks to session state."""
    st.session_state.processed_chunks[filename] = chunks


def get_processed_chunks(filename: str = None) -> Dict[str, List[DocumentChunk]]:
    """Get processed chunks from session state."""
    if filename:
        return st.session_state.processed_chunks.get(filename, [])
    return st.session_state.processed_chunks


def save_chunk_embeddings(filename: str, embeddings: List[Any]):
    """Save chunk embeddings to session state."""
    st.session_state.chunk_embeddings[filename] = embeddings


def get_chunk_embeddings(filename: str = None) -> Dict[str, List[Any]]:
    """Get chunk embeddings from session state."""
    if filename:
        return st.session_state.chunk_embeddings.get(filename, [])
    return st.session_state.chunk_embeddings


def store_processed_document(filename: str, document_data: Dict[str, Any]):
    """Store processed document data in session state."""
    st.session_state.processed_documents[filename] = document_data


def get_processed_document(filename: str) -> Optional[Dict[str, Any]]:
    """Retrieve processed document data from session state."""
    return st.session_state.processed_documents.get(filename)


def get_all_processed_documents() -> Dict[str, Dict[str, Any]]:
    """Get all processed documents from session state."""
    return st.session_state.processed_documents


def store_qa_data(qa_data: pd.DataFrame):
    """Store QA test set data in session state."""
    st.session_state.qa_data = qa_data


def get_qa_data() -> Optional[pd.DataFrame]:
    """Retrieve QA test set data from session state."""
    return st.session_state.qa_data


def clear_processed_documents():
    """Clear all processed documents from session state."""
    st.session_state.processed_documents = {}


def clear_qa_data():
    """Clear QA test set data from session state."""
    st.session_state.qa_data = None


def update_processing_options(*options):
    """Update document processing options in session state.
    
    Args:
        *options: Tuple of (param1, param2, splitter_type, vector_store_type) where:
            - For sentence splitter: (chunk_size, chunk_overlap, "sentence", vector_store_type)
            - For semantic splitter: (buffer_size, breakpoint_threshold, "semantic", vector_store_type)
    """
    param1, param2, splitter_type, vector_store_type = options
    st.session_state.splitter_type = splitter_type
    st.session_state.vector_store_type = vector_store_type
    
    if splitter_type == "sentence":
        st.session_state.sentence_options = {
            "chunk_size": param1,
            "chunk_overlap": param2
        }
    else:  # semantic
        st.session_state.semantic_options = {
            "buffer_size": param1,
            "breakpoint_threshold": param2
        }


def get_processing_options() -> tuple:
    """Get current processing options from session state.
    
    Returns:
        tuple: (param1, param2, splitter_type, vector_store_type) where:
            - For sentence splitter: (chunk_size, chunk_overlap, "sentence", vector_store_type)
            - For semantic splitter: (buffer_size, breakpoint_threshold, "semantic", vector_store_type)
    """
    splitter_type = st.session_state.splitter_type
    vector_store_type = st.session_state.vector_store_type
    
    if splitter_type == "sentence":
        options = st.session_state.sentence_options
        return (options["chunk_size"], options["chunk_overlap"], splitter_type, vector_store_type)
    else:  # semantic
        options = st.session_state.semantic_options
        return (options["buffer_size"], options["breakpoint_threshold"], splitter_type, vector_store_type)


def save_evaluation_results(results: Dict[str, Any]):
    """Save evaluation results to session state."""
    st.session_state.evaluation_results.update(results)


def get_evaluation_results() -> Dict[str, Any]:
    """Get evaluation results from session state."""
    return st.session_state.evaluation_results


def store_uploaded_file(file):
    """Store uploaded file in session state."""
    file_data = {
        "name": file.name,
        "type": file.type,
        "size": file.size,
        "content": file.getvalue()
    }
    st.session_state.uploaded_files[file.name] = file_data


def get_uploaded_files():
    """Get all uploaded files from session state.
    
    Returns:
        dict: Dictionary of file data indexed by filename
    """
    return st.session_state.uploaded_files


def clear_uploaded_files():
    """Clear all uploaded files from session state."""
    st.session_state.uploaded_files = {}


def clear_session_data():
    """Clear all session state data."""
    st.session_state.processed_chunks = {}
    st.session_state.chunk_embeddings = {}
    st.session_state.qa_data = None
    st.session_state.evaluation_results = {}
    st.session_state.uploaded_files = {}
