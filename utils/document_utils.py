from typing import List, Dict, Any, Literal
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.settings import Settings
from config import DEFAULT_BUFFER_SIZE, DEFAULT_BREAKPOINT_THRESHOLD, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
import faiss


# Define supported vector store types
VectorStoreType = Literal["simple", "faiss"]

# Define supported splitter types
SplitterType = Literal["sentence", "semantic"]


def extract_text_from_file(file) -> str:
    """Extract text from various file formats using LlamaIndex's SimpleDirectoryReader."""
    temp_dir = Path(tempfile.mkdtemp())
    temp_file_path = temp_dir / file.name

    try:
        with open(temp_file_path, "wb") as f:
            f.write(file.getvalue())

        reader = SimpleDirectoryReader(input_files=[temp_file_path])
        documents = reader.load_data()

        if documents:
            return documents[0].text
        return ""
    except Exception as e:
        print(f"Error extracting text from {file.name}: {e}")
        return ""
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_document(
    doc_in: Document, splitter_type: SplitterType = "sentence", **kwargs
) -> List[BaseNode]:
    """Process a LlamaIndex Document into a list of LlamaIndex Node objects (chunks).
    Args:
        doc_in: The LlamaIndex Document object to process.
        splitter_type: Type of splitter to use.
        **kwargs: Splitter-specific parameters.
    Returns:
        List[BaseNode]: A list of LlamaIndex Node objects.
    """
    if splitter_type == "sentence":
        chunk_size = kwargs.get("chunk_size")
        chunk_overlap = kwargs.get("chunk_overlap")
        
        if not chunk_size and not chunk_overlap:
            raise ValueError(
                "SentenceSplitter requires a chunk size and chunk overlap."
            )
        
        node_parser = SentenceSplitter.from_defaults(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    else: # splitter_type == "semantic"
        buffer_size = kwargs.get("buffer_size")
        breakpoint_threshold = kwargs.get("breakpoint_percentile_threshold")
        embed_model_override = kwargs.get("embed_model")

        if not Settings.embed_model and not embed_model_override and not buffer_size and not breakpoint_threshold:
            raise ValueError(
                "SemanticSplitter requires an embedding model, buffer size, and breakpoint threshold."
            )

        node_parser = SemanticSplitterNodeParser.from_defaults(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_threshold,
            embed_model=embed_model_override,  # Will use Settings.embed_model if None
        )
    nodes = node_parser.get_nodes_from_documents([doc_in])
    return nodes


def get_chunk_statistics(nodes: List[BaseNode]) -> Dict[str, Any]:
    """Get statistics about the chunks (LlamaIndex Nodes)."""
    if not nodes:
        return {
            "total_chunks": 0,
            "avg_chunk_size_chars": 0,
            "total_chars": 0,
            "avg_chunk_size_words": 0,
            "total_words": 0,
        }

    char_lengths = [len(node.get_content()) for node in nodes]
    word_counts = [len(node.get_content().split()) for node in nodes]

    return {
        "total_chunks": len(nodes),
        "avg_chunk_size_chars": sum(char_lengths) / len(nodes) if char_lengths else 0,
        "total_chars": sum(char_lengths),
        "avg_chunk_size_words": sum(word_counts) / len(nodes) if word_counts else 0,
        "total_words": sum(word_counts),
    }


def create_chunk_dataframe(nodes: List[BaseNode]) -> pd.DataFrame:
    """Create a DataFrame with chunk (LlamaIndex Node) information."""
    if not nodes:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "Node ID": node.node_id if node.node_id else f"node_{i}",
                "Text": node.get_content(),
                "Word Count": len(node.get_content().split()),
                "Char Count": len(node.get_content()),
                "Start Index": (
                    node.start_char_idx if node.start_char_idx is not None else "N/A"
                ),
                "End Index": (
                    node.end_char_idx if node.end_char_idx is not None else "N/A"
                ),
            }
            for i, node in enumerate(nodes)
        ]
    )


def create_index_from_documents(
    documents: List[Document], vector_store_type: VectorStoreType = "simple"
) -> VectorStoreIndex:
    """Create a vector store index from a list of LlamaIndex Documents."""
    if not Settings.embed_model:
        raise ValueError(
            "LlamaIndex Settings.embed_model is not configured. Cannot create index. "
            "Ensure API key and model settings are correctly configured."
        )

    if vector_store_type == "simple":
        vector_store = SimpleVectorStore()
    else: # vector_store_type == "faiss"
        dimension = None
        if hasattr(Settings.embed_model, "embed_dim") and Settings.embed_model.embed_dim is not None:
            dimension = Settings.embed_model.embed_dim
        # Fallback for known model names if embed_dim is not directly available
        elif hasattr(Settings.embed_model, "model_name"):
            model_name_from_settings = Settings.embed_model.model_name
            model_dimension_map = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            if model_name_from_settings in model_dimension_map:
                dimension = model_dimension_map[model_name_from_settings]
            else:
                print(f"Warning: FAISS dimension fallback for model '{model_name_from_settings}' not found. Attempting to infer dimension.")
                # Attempt to infer dimension
                try:
                    dummy_embedding = Settings.embed_model.get_text_embedding("test")
                    dimension = len(dummy_embedding)
                    print(f"Inferred FAISS dimension as {dimension} for model '{model_name_from_settings}'.")
                except Exception as e:
                    raise ValueError(
                        f"Could not determine embedding dimension for FAISS. Model '{model_name_from_settings}' "
                        f"does not have 'embed_dim', is not in known mappings, and failed to infer dimension: {e}. "
                        f"Current embed_model type: {type(Settings.embed_model).__name__}."
                    )

        if dimension is None:
            # This case should ideally be caught by the logic above.
            raise ValueError(
                "Could not determine embedding dimension for FAISS from LlamaIndex Settings.embed_model. "
                f"Ensure Settings.embed_model (current type: {type(Settings.embed_model).__name__}) has an 'embed_dim' attribute, a recognized 'model_name', or supports inference."
            )

        faiss_index = faiss.IndexFlatL2(dimension)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    return index