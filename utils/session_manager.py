import streamlit as st
from typing import List, Dict, Any, Optional
import pandas as pd

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
import json

class SessionManager:
    def __init__(self):
        self._initialize_session_state()

    def _init_key_if_not_exists(self, key: str, value: Any):
        """Helper to initialize a session state key if it doesn't exist."""
        if key not in st.session_state:
            st.session_state[key] = value
    
    def _initialize_session_state(self):
        """Initialize all session state variables with default values."""

        # --- Core Application Data ---
        self._init_key_if_not_exists("processed_documents", []) # Stores List[Dict[str, Any]] (processed doc data)
        self._init_key_if_not_exists("uploaded_files", [])     # Stores List[UploadedFile] from Streamlit
        self._init_key_if_not_exists("qa_data", None)         # Stores Optional[pd.DataFrame]
        self._init_key_if_not_exists("index", None)           # Stores LlamaIndex VectorStoreIndex or similar

        # --- API Key Management ---
        self._init_key_if_not_exists("api_keys", {"openai": ""}) # Internal storage for actual keys
        self._init_key_if_not_exists("api_key_valid", False)    # Boolean flag for UI
        # Note: "openai_key" (the widget key for st.text_input) will be implicitly created by Streamlit
        # or you can initialize it too if you want to ensure it exists before widget rendering, e.g.:
        self._init_key_if_not_exists("openai_key", "") # For the API key input widget

        # --- Model Settings (Direct Widget Keys & Internal Structured Storage) ---
        # Direct keys for UI widgets (these are what Home.py selectboxes will use for their 'key' and read for 'value'/'index')
        self._init_key_if_not_exists("llm_model", DEFAULT_LLM_MODEL)
        self._init_key_if_not_exists("embedding_model", DEFAULT_EMBEDDING_MODEL)
        # Internal structured storage for model settings (e.g., provider, which might not have a direct widget)
        self._init_key_if_not_exists("model_settings_internal", {
            "embedding_model": DEFAULT_EMBEDDING_MODEL, # Can be redundant if direct keys are primary
            "llm_model": DEFAULT_LLM_MODEL,           # Can be redundant
            "llm_provider": DEFAULT_LLM_PROVIDER      # This might be unique here
        })

        # --- Processing Options (Direct Widget Keys & Internal Structured Storage) ---
        # Direct keys for UI widgets
        self._init_key_if_not_exists("splitter_type", DEFAULT_SPLITTER_TYPE)
        self._init_key_if_not_exists("vector_store_type", DEFAULT_VECTOR_STORE_TYPE)
        self._init_key_if_not_exists("chunk_size", DEFAULT_CHUNK_SIZE)
        self._init_key_if_not_exists("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self._init_key_if_not_exists("buffer_size", DEFAULT_BUFFER_SIZE)
        self._init_key_if_not_exists("breakpoint_threshold", DEFAULT_BREAKPOINT_THRESHOLD)
        # Internal structured storage for options (useful for grouping in get/load_app_configuration)
        self._init_key_if_not_exists("sentence_options_internal", {
            "chunk_size": DEFAULT_CHUNK_SIZE,         # Can be redundant
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP    # Can be redundant
        })
        self._init_key_if_not_exists("semantic_options_internal", { # Corrected from "semantic_options" to avoid potential name clash if used as direct key
            "buffer_size": DEFAULT_BUFFER_SIZE,       # Can be redundant
            "breakpoint_threshold": DEFAULT_BREAKPOINT_THRESHOLD # Can be redundant
        })
        
        # --- Chat History ---
        self._init_key_if_not_exists("chat_history", [])

        # --- Evaluation Runs ---
        # Stores a list of dictionaries, each representing an evaluation run.
        self._init_key_if_not_exists("evaluation_runs", []) 

        # --- UI Control Flags (Example from previous discussions) ---
        if "config_just_loaded" not in st.session_state: # Ensure this exists if Home.py uses it
            st.session_state.config_just_loaded = False

        # --- Chunk Embeddings (Assess its use) ---
        # Kept for now; remove if unused.
        self._init_key_if_not_exists("chunk_embeddings", {})

    # --- Configuration Management ---
    def get_app_configuration(self) -> Dict[str, Any]:
        """
        Gathers all relevant application configurations into a dictionary.
        The API key is explicitly excluded for security.
        Returns:
            Dict[str, Any]: A dictionary containing the application's configuration.
        """
        config = {
            "app_version": "1.0",
            "model_settings": self.get_model_settings().copy(),
            "processing_options": {
                "splitter_type": self.splitter_type,
                "vector_store_type": self.vector_store_type,
                "sentence_options": st.session_state.sentence_options_internal.copy(),
                "semantic_options": st.session_state.semantic_options_internal.copy(),
            }
        }
        return config

    def load_app_configuration(self, config_data: Dict[str, Any]):
        loaded_model_settings = config_data.get("model_settings")
        if loaded_model_settings:
            # Use the SM setters which will update the st.session_state.widget_key
            self.embedding_model = loaded_model_settings.get("embedding_model", self.embedding_model)
            self.llm_model = loaded_model_settings.get("llm_model", self.llm_model)
            # Provider still goes to internal dict if no widget for it
            st.session_state.model_settings_internal["llm_provider"] = loaded_model_settings.get("llm_provider", 
                st.session_state.model_settings_internal["llm_provider"])


        loaded_processing_options = config_data.get("processing_options")
        if loaded_processing_options:
            self.splitter_type = loaded_processing_options.get("splitter_type", self.splitter_type)
            self.vector_store_type = loaded_processing_options.get("vector_store_type", self.vector_store_type)
            
            sentence_opts = loaded_processing_options.get("sentence_options", st.session_state.sentence_options_internal)
            semantic_opts = loaded_processing_options.get("semantic_options", st.session_state.semantic_options_internal)

            # Update internal dicts if still used
            st.session_state.sentence_options_internal = sentence_opts.copy()
            st.session_state.semantic_options_internal = semantic_opts.copy()

            if self.splitter_type == "sentence":
                self.chunk_size = sentence_opts.get("chunk_size", DEFAULT_CHUNK_SIZE)
                self.chunk_overlap = sentence_opts.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            else: # semantic
                self.buffer_size = semantic_opts.get("buffer_size", DEFAULT_BUFFER_SIZE)
                self.breakpoint_threshold = semantic_opts.get("breakpoint_threshold", DEFAULT_BREAKPOINT_THRESHOLD)
        
        print("Configuration loaded.")


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

    # --- API Key Settings ---
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
        
    # --- Model Settings ---
    @property
    def llm_model(self) -> str:
        return st.session_state.llm_model # Read from direct widget key

    @llm_model.setter
    def llm_model(self, value: str):
        st.session_state.llm_model = value # Set direct widget key
        st.session_state.model_settings_internal["llm_model"] = value # Update internal store

    @property
    def embedding_model(self) -> str:
        return st.session_state.embedding_model

    @embedding_model.setter
    def embedding_model(self, value: str):
        st.session_state.embedding_model = value
        st.session_state.model_settings_internal["embedding_model"] = value
    
    def get_model_settings(self) -> Dict[str, str]: # This can return the internal dict
        return st.session_state.model_settings_internal

    def update_model_settings(self, embedding_model: Optional[str] = None,
                              llm_model: Optional[str] = None,
                              llm_provider: Optional[str] = None):
        if embedding_model is not None:
            self.embedding_model = embedding_model # Use setter
        if llm_model is not None:
            self.llm_model = llm_model # Use setter
        if llm_provider is not None: # Provider might not have a direct widget
            st.session_state.model_settings_internal["llm_provider"] = llm_provider

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
        return st.session_state.sentence_options_internal["chunk_size"]

    @chunk_size.setter
    def chunk_size(self, value: int):
        st.session_state.sentence_options_internal["chunk_size"] = value

    @property
    def chunk_overlap(self) -> int:
        return st.session_state.sentence_options_internal["chunk_overlap"]

    @chunk_overlap.setter
    def chunk_overlap(self, value: int):
        st.session_state.sentence_options_internal["chunk_overlap"] = value

    # --- Semantic Splitter Options ---
    @property
    def buffer_size(self) -> int:
        return st.session_state.semantic_options_internal["buffer_size"]

    @buffer_size.setter
    def buffer_size(self, value: int):
        st.session_state.semantic_options_internal["buffer_size"] = value

    @property
    def breakpoint_threshold(self) -> int:
        return st.session_state.semantic_options_internal["breakpoint_threshold"]

    @breakpoint_threshold.setter
    def breakpoint_threshold(self, value: int):
        st.session_state.semantic_options_internal["breakpoint_threshold"] = value

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

    # --- Evaluation Run Management ---
    def add_evaluation_run(self, run_data: Dict[str, Any]):
        """
        Adds a new evaluation run to the session.
        Args:
            run_data (Dict[str, Any]): A dictionary containing details of the run,
                                       e.g., name, timestamp, config_summary, results_df.
        """
        st.session_state.evaluation_runs.append(run_data)

    def get_all_evaluation_runs(self) -> List[Dict[str, Any]]:
        """Gets all stored evaluation runs."""
        return st.session_state.get("evaluation_runs", [])

    def get_evaluation_run_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Gets a specific evaluation run by its name."""
        for run in st.session_state.get("evaluation_runs", []):
            if run.get("name") == name:
                return run
        return None

    def clear_all_evaluation_runs(self):
        """Clears all stored evaluation runs."""
        st.session_state.evaluation_runs = []

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
        self.clear_all_evaluation_runs()
        self.clear_chunk_embeddings()

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
        st.session_state.sentence_options_internal = {
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "chunk_overlap": DEFAULT_CHUNK_OVERLAP
        }
        st.session_state.semantic_options_internal = {
            "buffer_size": DEFAULT_BUFFER_SIZE,
            "breakpoint_threshold": DEFAULT_BREAKPOINT_THRESHOLD
        }
        st.toast("Session data cleared and settings reset to default.", icon="🧹")