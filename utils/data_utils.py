from typing import List, Dict, Any, Optional
import streamlit as st
from pathlib import Path
import pandas as pd
from utils.document_utils import DocumentChunk
from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)


def initialize_session_state():
    """Initialize all session state variables used across the application."""
    # Document processing state
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = {}

    # QA test set state
    if "qa_data" not in st.session_state:
        st.session_state.qa_data = None

    # Processing options state
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP


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


def update_processing_options(chunk_size: int, chunk_overlap: int):
    """Update document processing options in session state."""
    st.session_state.chunk_size = chunk_size
    st.session_state.chunk_overlap = chunk_overlap


def get_processing_options() -> tuple[int, int]:
    """Get current processing options from session state."""
    return st.session_state.chunk_size, st.session_state.chunk_overlap


def save_evaluation_results(results: Dict[str, Any]):
    """Save evaluation results to session state."""
    st.session_state.evaluation_results.update(results)


def get_evaluation_results() -> Dict[str, Any]:
    """Get evaluation results from session state."""
    return st.session_state.evaluation_results


def clear_session_data():
    """Clear all session state data."""
    st.session_state.processed_chunks = {}
    st.session_state.chunk_embeddings = {}
    st.session_state.qa_data = None
    st.session_state.evaluation_results = {}
