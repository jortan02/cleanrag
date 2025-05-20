import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="CleanRAG - Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("Settings & Configuration ⚙️")

# Model Configuration
st.header("Model Configuration")

# Local vs Cloud Mode
st.subheader("Processing Mode")
processing_mode = st.radio(
    "Select Processing Mode",
    ["Local", "Cloud"],
    horizontal=True,
    help="Choose between local processing (HuggingFace models) or cloud-based processing"
)

if processing_mode == "Local":
    # Local Model Settings
    st.subheader("Local Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Embedding Model")
        embedding_model = st.selectbox(
            "Select Embedding Model",
            [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "BAAI/bge-small-en-v1.5"
            ],
            help="Models will be downloaded from HuggingFace Hub"
        )
        
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=500,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=200,
            value=50,
            step=10,
            help="Overlap between consecutive chunks"
        )
    
    with col2:
        st.markdown("### LLM Settings")
        llm_model = st.selectbox(
            "Select LLM Model",
            [
                "mistralai/Mistral-7B-v0.1",
                "meta-llama/Llama-2-7b",
                "mosaicml/mpt-7b"
            ],
            help="Models will be downloaded from HuggingFace Hub"
        )
        
        context_window = st.slider(
            "Context Window",
            min_value=512,
            max_value=8192,
            value=4096,
            step=512,
            help="Maximum context window size for the LLM"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=2048,
            value=512,
            step=128,
            help="Maximum number of tokens to generate"
        )

else:
    # Cloud Settings
    st.subheader("Cloud Settings")
    
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Your API key for cloud services"
    )
    
    api_endpoint = st.text_input(
        "API Endpoint",
        value="https://api.example.com/v1",
        help="Endpoint for cloud API"
    )

# Retrieval Settings
st.header("Retrieval Settings")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Search Configuration")
    search_mode = st.selectbox(
        "Search Mode",
        ["Semantic", "Hybrid", "Keyword"],
        help="Method used for document retrieval"
    )
    
    similarity_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum similarity score for retrieval"
    )

with col2:
    st.markdown("### Reranking Settings")
    enable_reranking = st.checkbox(
        "Enable Reranking",
        value=True,
        help="Enable reranking of retrieved results"
    )
    
    if enable_reranking:
        reranking_model = st.selectbox(
            "Reranking Model",
            ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"],
            help="Models will be downloaded from HuggingFace Hub"
        )
        
        reranking_threshold = st.slider(
            "Reranking Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

# Export Settings
st.header("Export Settings")
export_format = st.selectbox(
    "Export Format",
    ["JSON", "CSV", "PDF"]
)

if st.button("Save Configuration", type="primary"):
    # Create configuration dictionary
    config = {
        "processing_mode": processing_mode,
        "embedding_model": embedding_model if processing_mode == "Local" else None,
        "chunk_size": chunk_size if processing_mode == "Local" else None,
        "chunk_overlap": chunk_overlap if processing_mode == "Local" else None,
        "llm_model": llm_model if processing_mode == "Local" else None,
        "context_window": context_window if processing_mode == "Local" else None,
        "max_tokens": max_tokens if processing_mode == "Local" else None,
        "api_key": api_key if processing_mode == "Cloud" else None,
        "api_endpoint": api_endpoint if processing_mode == "Cloud" else None,
        "search_mode": search_mode,
        "similarity_threshold": similarity_threshold,
        "enable_reranking": enable_reranking,
        "reranking_model": reranking_model if enable_reranking else None,
        "reranking_threshold": reranking_threshold if enable_reranking else None,
        "export_format": export_format
    }
    
    # Save configuration
    config_path = Path("config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    st.success("Configuration saved successfully!") 