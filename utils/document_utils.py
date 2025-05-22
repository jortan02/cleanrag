from typing import List, Dict, Any, Tuple, Optional, Literal
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
import tempfile
import shutil
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    StorageContext,
)
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os
import re
import streamlit as st
from llama_index.embeddings.openai import OpenAIEmbedding
from utils.api_config import get_api_key, get_model_settings


# Define supported vector store types
VectorStoreType = Literal["simple", "faiss"]  # Can be extended with more options

# Define supported splitter types
SplitterType = Literal["sentence", "semantic"]


@dataclass
class DocumentChunk:
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: Dict[str, Any] = None


def extract_text_from_file(file) -> str:
    """Extract text from various file formats using LlamaIndex's SimpleDirectoryReader."""
    # Create a unique temporary directory for this file
    temp_dir = Path(tempfile.mkdtemp())
    temp_path = temp_dir / file.name

    try:
        # Save the uploaded file temporarily
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        # Use LlamaIndex to extract text
        documents = SimpleDirectoryReader(str(temp_dir)).load_data()
        if documents:
            return documents[0].text
        return ""
    finally:
        # Clean up temporary directory and its contents
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_document(
    text: str, splitter_type: SplitterType = "sentence", **kwargs
) -> List[DocumentChunk]:
    """Process document text into chunks using specified splitter type.

    Args:
        text: The text to process
        splitter_type: Type of splitter to use
        **kwargs: Splitter-specific parameters:
            - For sentence-based splitting:
                - chunk_size: Target size for each chunk (default: 500)
                - chunk_overlap: Number of tokens to overlap between chunks (default: 50)
            - For semantic splitting:
                - buffer_size: Number of sentences to buffer (default: 1)
                - breakpoint_percentile_threshold: Percentile threshold for breakpoints (default: 95)
    """
    # Create a document
    doc = Document(text=text)

    if splitter_type == "sentence":
        # Extract sentence splitter specific parameters
        chunk_size = kwargs.pop("chunk_size", 500)
        chunk_overlap = kwargs.pop("chunk_overlap", 50)
        node_parser = SentenceSplitter.from_defaults(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else:  # semantic splitting
        # Extract semantic splitter specific parameters
        buffer_size = kwargs.pop("buffer_size", 1)
        breakpoint_threshold = kwargs.pop("breakpoint_percentile_threshold", 95)
        node_parser = SemanticSplitterNodeParser.from_defaults(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_threshold,
        )

    nodes = node_parser.get_nodes_from_documents([doc])

    # Convert nodes to DocumentChunks
    chunks = []
    for i, node in enumerate(nodes):
        chunks.append(
            DocumentChunk(
                text=node.text,
                start_idx=node.start_char_idx if hasattr(node, "start_char_idx") else 0,
                end_idx=(
                    node.end_char_idx
                    if hasattr(node, "end_char_idx")
                    else len(node.text)
                ),
                chunk_id=i,
                metadata=node.metadata,
            )
        )

    return chunks


def get_chunk_statistics(chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Get statistics about the chunks."""
    return {
        "total_chunks": len(chunks),
        "avg_chunk_size": (
            sum(len(chunk.text.split()) for chunk in chunks) / len(chunks)
            if chunks
            else 0
        ),
        "total_words": sum(len(chunk.text.split()) for chunk in chunks),
        "chunk_sizes": [len(chunk.text.split()) for chunk in chunks],
    }


def create_chunk_dataframe(chunks: List[DocumentChunk]) -> pd.DataFrame:
    """Create a DataFrame with chunk information."""
    return pd.DataFrame(
        [
            {
                "Chunk ID": chunk.chunk_id,
                "Text": chunk.text,
                "Word Count": len(chunk.text.split()),
                "Start Position": chunk.start_idx,
                "End Position": chunk.end_idx,
                "Metadata": str(chunk.metadata) if chunk.metadata else None,
            }
            for chunk in chunks
        ]
    )


def create_index_from_documents(
    documents: List[Document], vector_store_type: str = "simple"
) -> VectorStoreIndex:
    """Create a vector store index from a list of documents.

    Args:
        documents: List of LlamaIndex documents
        vector_store_type: Type of vector store to use ("simple" or "faiss")

    Returns:
        VectorStoreIndex: The created index
    """
    # Get model settings
    settings = get_model_settings()

    # Create vector store
    if vector_store_type == "simple":
        vector_store = SimpleVectorStore()
    else:  # faiss
        dimension = 384  # Default for MiniLM
        if settings["embedding_model"] == "text-embedding-3-small":
            dimension = 1536
        elif settings["embedding_model"] == "text-embedding-3-large":
            dimension = 3072
        elif settings["embedding_model"] == "all-mpnet-base-v2":
            dimension = 768
        vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dimension))

    # Create and return index
    return VectorStoreIndex.from_documents(
        documents,
        vector_store=vector_store,
    )
