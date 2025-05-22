import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd
# from streamlit.runtime.uploaded_file_manager import UploadedFile # Specific type for uploaded files
# from llama_index.core.schema import BaseNode # For stricter typing if needed

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_BUFFER_SIZE,
    DEFAULT_BREAKPOINT_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_SPLITTER_TYPE,
    DEFAULT_VECTOR_STORE_TYPE
)

class SessionManager:
    def __init__(self):
        self._initialize_session_state()

    def _init_key_if_not_exists(self, key: str, value: Any):
        """Helper to initialize a session state key if it doesn't exist."""
        if key not in st.session_state:
            st.session_state[key] = value

    def _initialize_session_state(self):
        """Initialize all session state variables with default values."""
        # --- Core Data ---
        # "processed_documents" stores a list of dictionaries. Each dictionary represents a processed document
        # and contains various keys, including 'name', 'text', 'document' (LlamaIndex Document object),
        # and 'chunks' (which is a List of LlamaIndex Node objects).
        self._init_key_if_not_exists("processed_documents", [])
        self._init_key_if_not_exists("uploaded_files", []) # Stores Streamlit UploadedFile objects
        self._init_key_if_not_exists("qa_data", None) # Stores pandas DataFrame
        self._init_key_if_not_exists("index", None) # Stores LlamaIndex VectorStoreIndex or similar

        # --- API and Model Settings ---
        self._init_key_if_not_exists("api_keys", {"openai": ""})
        self._init_key_if_not_exists("model_settings", {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "llm_model": DEFAULT_LLM_MODEL,
            "llm_provider": DEFAULT_LLM_PROVIDER
        })
        self._init_key_if_not_exists("api_key_valid", False)

        # --- Processing Options ---
        self._init_key_if_not_exists("splitter_type", DEFAULT_SPLITTER_TYPE)
        self._init_key_if_not_exists("vector_store_type", DEFAULT_VECTOR_STORE_TYPE)
        self._init_key_if_not_exists("sentence_options", {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP
        })
        self._init_key_if_not_exists("semantic_options", {
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "breakpoint_threshold": DEFAULT_BREAKPOINT_THRESHOLD
        })

        # --- Chat ---
        self._init_key_if_not_exists("chat_history", [])

        # --- Evaluation ---
        self._init_key_if_not_exists("evaluation_results", {})
        
        # --- Chunk Embeddings (Assess if this is still needed/how it's used) ---
        # If embeddings are stored per file and tied to nodes, this might be relevant.
        # If it was tied to the old DocumentChunk structure, it needs re-evaluation.
        self._init_key_if_not_exists("chunk_embeddings", {}) # Dict[filename, List[embeddings]]


    # --- Processed Documents ---
    def store_processed_document(self, document_data: Dict[str, Any]):
        """
        Stores data for a single processed document.
        The 'document_data' dictionary is expected to contain:
        - 'name': str (filename)
        - 'text': str (full text of the document)
        - 'document': llama_index.core.Document (the LlamaIndex Document object)
        - 'chunks': List[BaseNode] (List of LlamaIndex Node objects from parsing)
        - 'stats': Dict (statistics about the chunks/nodes)
        - 'chunk_df': pd.DataFrame (DataFrame representation of chunks/nodes)
        """
        st.session_state.processed_documents.append(document_data)

    def get_all_processed_documents(self) -> List[Dict[str, Any]]:
        """
        Gets all processed document data. Each item in the list is a dictionary
        as described in store_processed_document, where 'chunks' holds a List of
        LlamaIndex Node objects.
        """
        return st.session_state.processed_documents

    def clear_processed_documents(self):
        st.session_state.processed_documents = []
        self.index = None # Also clear the associated index

    # --- Uploaded Files ---
    def store_uploaded_files(self, files: List[Any]): # files are Streamlit's UploadedFile objects
        st.session_state.uploaded_files = files

    def get_uploaded_files(self) -> List[Any]: # Returns List[UploadedFile]
        return st.session_state.uploaded_files

    def clear_uploaded_files(self):
        st.session_state.uploaded_files = []

    # --- QA Data ---
    def store_qa_data(self, qa_data: Optional[pd.DataFrame]):
        st.session_state.qa_data = qa_data

    def get_qa_data(self) -> Optional[pd.DataFrame]:
        return st.session_state.qa_data

    def clear_qa_data(self):
        st.session_state.qa_data = None

    # --- Index ---
    @property
    def index(self) -> Any: # Replace Any with actual index type e.g. VectorStoreIndex
        return st.session_state.get("index") # Use .get for graceful None if not set

    @index.setter
    def index(self, value: Any):
        st.session_state.index = value

    # --- API Keys and Model Settings ---
    def get_api_key(self, provider: str = "openai") -> str:
        return st.session_state.api_keys.get(provider, "")

    def update_api_key(self, key: str, provider: str = "openai"):
        st.session_state.api_keys[provider] = key

    @property
    def api_key_valid(self) -> bool:
        return st.session_state.api_key_valid

    @api_key_valid.setter
    def api_key_valid(self, value: bool):
        st.session_state.api_key_valid = value

    def get_model_settings(self) -> Dict[str, str]:
        return st.session_state.model_settings

    def update_model_settings(self, embedding_model: Optional[str] = None,
                              llm_model: Optional[str] = None,
                              llm_provider: Optional[str] = None):
        if embedding_model is not None:
            st.session_state.model_settings["embedding_model"] = embedding_model
        if llm_model is not None:
            st.session_state.model_settings["llm_model"] = llm_model
        if llm_provider is not None:
            st.session_state.model_settings["llm_provider"] = llm_provider

    @property
    def embedding_model(self) -> str:
        return st.session_state.model_settings["embedding_model"]

    @embedding_model.setter
    def embedding_model(self, value: str):
        st.session_state.model_settings["embedding_model"] = value

    @property
    def llm_model(self) -> str:
        return st.session_state.model_settings["llm_model"]

    @llm_model.setter
    def llm_model(self, value: str):
        st.session_state.model_settings["llm_model"] = value


    # --- Processing Options ---
    @property
    def splitter_type(self) -> str:
        return st.session_state.splitter_type

    @splitter_type.setter
    def splitter_type(self, value: str):
        st.session_state.splitter_type = value

    @property
    def vector_store_type(self) -> str:
        return st.session_state.vector_store_type

    @vector_store_type.setter
    def vector_store_type(self, value: str):
        st.session_state.vector_store_type = value

    # --- Sentence Splitter Options ---
    @property
    def chunk_size(self) -> int:
        return st.session_state.sentence_options["chunk_size"]

    @chunk_size.setter
    def chunk_size(self, value: int):
        st.session_state.sentence_options["chunk_size"] = value

    @property
    def chunk_overlap(self) -> int:
        return st.session_state.sentence_options["chunk_overlap"]

    @chunk_overlap.setter
    def chunk_overlap(self, value: int):
        st.session_state.sentence_options["chunk_overlap"] = value

    # --- Semantic Splitter Options ---
    @property
    def buffer_size(self) -> int:
        return st.session_state.semantic_options["buffer_size"]

    @buffer_size.setter
    def buffer_size(self, value: int):
        st.session_state.semantic_options["buffer_size"] = value

    @property
    def breakpoint_threshold(self) -> int:
        return st.session_state.semantic_options["breakpoint_threshold"]

    @breakpoint_threshold.setter
    def breakpoint_threshold(self, value: int):
        st.session_state.semantic_options["breakpoint_threshold"] = value

    # --- Combined Getter/Setter for all processing options ---
    def update_processing_options(self, param1: int, param2: int, splitter_type: str, vector_store_type: str):
        self.splitter_type = splitter_type
        self.vector_store_type = vector_store_type

        if splitter_type == "sentence":
            self.chunk_size = param1
            self.chunk_overlap = param2
        elif splitter_type == "semantic":
            self.buffer_size = param1
            self.breakpoint_threshold = param2
        else:
            st.warning(f"Unknown splitter type: {splitter_type} in update_processing_options")


    def get_processing_options(self) -> tuple:
        current_splitter_type = self.splitter_type
        current_vector_store_type = self.vector_store_type

        if current_splitter_type == "sentence":
            return (self.chunk_size, self.chunk_overlap, current_splitter_type, current_vector_store_type)
        elif current_splitter_type == "semantic":
            return (self.buffer_size, self.breakpoint_threshold, current_splitter_type, current_vector_store_type)
        else:
            st.warning(f"Unknown splitter type: {current_splitter_type} in get_processing_options. Falling back to sentence defaults.")
            return (DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, "sentence", current_vector_store_type)


    # --- Chat History ---
    def get_session_chat_history(self) -> List[str]:
        return st.session_state.chat_history

    def update_session_chat_history(self, message: str):
        st.session_state.chat_history.append(message)

    def clear_session_chat_history(self):
        st.session_state.chat_history = []

    # --- Evaluation Results ---
    def save_evaluation_results(self, results: Dict[str, Any]):
        st.session_state.evaluation_results.update(results)

    def get_evaluation_results(self) -> Dict[str, Any]:
        return st.session_state.evaluation_results

    def clear_evaluation_results(self):
        st.session_state.evaluation_results = {}

    # --- Chunk Embeddings Storage (Assess its use) ---
    # If you have a separate process for generating and storing embeddings per file/node,
    # these methods might still be useful. Otherwise, they can be removed.
    def save_chunk_embeddings(self, filename: str, embeddings: List[Any]):
        st.session_state.chunk_embeddings[filename] = embeddings

    def get_chunk_embeddings(self, filename: Optional[str] = None) -> Any:
        if filename:
            return st.session_state.chunk_embeddings.get(filename, [])
        return st.session_state.chunk_embeddings

    def clear_chunk_embeddings(self, filename: Optional[str] = None):
        if filename and filename in st.session_state.chunk_embeddings:
            del st.session_state.chunk_embeddings[filename]
        elif filename is None:
            st.session_state.chunk_embeddings = {}


    # --- Clear All Session Data ---
    def clear_all_session_data(self):
        """Clears most user-generated data from the session, resetting to a near-initial state."""
        self.clear_processed_documents()
        self.clear_uploaded_files()
        self.clear_qa_data()
        self.clear_session_chat_history()
        self.clear_evaluation_results()
        self.clear_chunk_embeddings() # Clear this too if it's being used

        # Reset settings to their initial defaults from config
        st.session_state.api_keys = {"openai": ""}
        st.session_state.model_settings = {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "llm_model": DEFAULT_LLM_MODEL,
            "llm_provider": DEFAULT_LLM_PROVIDER
        }
        st.session_state.api_key_valid = False
        st.session_state.splitter_type = DEFAULT_SPLITTER_TYPE
        st.session_state.vector_store_type = DEFAULT_VECTOR_STORE_TYPE
        st.session_state.sentence_options = {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP
        }
        st.session_state.semantic_options = {
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "breakpoint_threshold": DEFAULT_BREAKPOINT_THRESHOLD
        }
        st.toast("Session data cleared and settings reset to default.", icon="🧹")