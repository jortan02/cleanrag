from typing import List, Dict, Any
import streamlit as st
from pathlib import Path
import pandas as pd
from utils.document_utils import DocumentChunk


def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if "processed_chunks" not in st.session_state:
        st.session_state.processed_chunks = (
            {}
        )  # Dict of filename -> List[DocumentChunk]

    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = {}  # Dict of filename -> List[embeddings]

    if "qa_dataset" not in st.session_state:
        st.session_state.qa_dataset = None  # DataFrame for QA test set

    if "evaluation_results" not in st.session_state:
        st.session_state.evaluation_results = {}  # Dict to store evaluation metrics


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


def save_qa_dataset(qa_data: pd.DataFrame):
    """Save QA dataset to session state."""
    st.session_state.qa_dataset = qa_data


def get_qa_dataset() -> pd.DataFrame:
    """Get QA dataset from session state."""
    return st.session_state.qa_dataset


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
    st.session_state.qa_dataset = None
    st.session_state.evaluation_results = {}
