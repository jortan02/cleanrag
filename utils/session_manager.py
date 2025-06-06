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
        self._init_key_if_not_exists("processed_documents", [])
        self._init_key_if_not_exists("uploaded_files", [])
        self._init_key_if_not_exists("qa_data", None)
        self._init_key_if_not_exists("index", None)

        # --- API Key Management ---
        self._init_key_if_not_exists("api_keys_internal", {"openai": ""})
        self._init_key_if_not_exists("api_key_valid", False)
        self._init_key_if_not_exists("openai_key", self.get_api_key("openai"))
        
        # --- Configuration Management ---
        self._init_key_if_not_exists("loaded_config_data_state", None)
        self._init_key_if_not_exists("last_uploaded_config_id", None)

        # --- Model Settings (Direct Widget Keys) ---
        self._init_key_if_not_exists("llm_model", DEFAULT_LLM_MODEL)
        self._init_key_if_not_exists("embedding_model", DEFAULT_EMBEDDING_MODEL)
        self._init_key_if_not_exists("llm_provider_internal", DEFAULT_LLM_PROVIDER)

        # --- Processing Options (Direct Widget Keys) ---
        self._init_key_if_not_exists("splitter_type", DEFAULT_SPLITTER_TYPE)
        self._init_key_if_not_exists("vector_store_type", DEFAULT_VECTOR_STORE_TYPE)
        self._init_key_if_not_exists("chunk_size", DEFAULT_CHUNK_SIZE)
        self._init_key_if_not_exists("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self._init_key_if_not_exists("buffer_size", DEFAULT_BUFFER_SIZE)
        self._init_key_if_not_exists("breakpoint_threshold", DEFAULT_BREAKPOINT_THRESHOLD)

        # --- Chat History ---
        self._init_key_if_not_exists("chat_history", [])

        # --- Evaluation Runs ---
        self._init_key_if_not_exists("evaluation_runs", [])
        
    # --- API Key Settings ---
    @property
    def openai_key_widget_value(self) -> str:
        return st.session_state.openai_key

    @openai_key_widget_value.setter
    def openai_key_widget_value(self, value: str):
        st.session_state.openai_key = value

    def get_api_key(self, provider: str = "openai") -> str:
        return st.session_state.api_keys_internal.get(provider, "")

    def update_api_key(self, key: str, provider: str = "openai"):
        st.session_state.api_keys_internal[provider] = key

    # --- Configuration Management ---
    @property
    def loaded_config_data_state(self) -> Dict[str, Any]:
        return st.session_state.loaded_config_data_state

    @loaded_config_data_state.setter
    def loaded_config_data_state(self, value: Dict[str, Any]):
        st.session_state.loaded_config_data_state = value
        
    @property
    def last_uploaded_config_id(self) -> str:
        return st.session_state.last_uploaded_config_id

    @last_uploaded_config_id.setter
    def last_uploaded_config_id(self, value: str):
        st.session_state.last_uploaded_config_id = value

    def get_app_configuration(self) -> Dict[str, Any]:
        """
        Gathers all relevant application configurations into a dictionary.
        The API key is explicitly excluded for security.
        Returns:
            Dict[str, Any]: A dictionary containing the application's configuration.
        """
        config = {
            "app_version": "1.0",
            "model_settings": {
                "embedding_model": self.embedding_model,
                "llm_model": self.llm_model,
                "llm_provider": self.llm_provider
            },
            "processing_options": {
                "splitter_type": self.splitter_type,
                "vector_store_type": self.vector_store_type,
                # Store current values based on active splitter
                "sentence_options": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                },
                "semantic_options": {
                    "buffer_size": self.buffer_size,
                    "breakpoint_threshold": self.breakpoint_threshold
                },
            }
        }
        return config

    def load_app_configuration(self, config_data: Dict[str, Any]):
        loaded_model_settings = config_data.get("model_settings")
        if loaded_model_settings:
            self.embedding_model = loaded_model_settings.get("embedding_model", DEFAULT_EMBEDDING_MODEL)
            self.llm_model = loaded_model_settings.get("llm_model", DEFAULT_LLM_MODEL)
            self.llm_provider = loaded_model_settings.get("llm_provider", DEFAULT_LLM_PROVIDER)

        loaded_processing_options = config_data.get("processing_options")
        if loaded_processing_options:
            self.splitter_type = loaded_processing_options.get("splitter_type", DEFAULT_SPLITTER_TYPE)
            self.vector_store_type = loaded_processing_options.get("vector_store_type", DEFAULT_VECTOR_STORE_TYPE)

            sentence_opts = loaded_processing_options.get("sentence_options", {})
            self.chunk_size = sentence_opts.get("chunk_size", DEFAULT_CHUNK_SIZE)
            self.chunk_overlap = sentence_opts.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)

            semantic_opts = loaded_processing_options.get("semantic_options", {})
            self.buffer_size = semantic_opts.get("buffer_size", DEFAULT_BUFFER_SIZE)
            self.breakpoint_threshold = semantic_opts.get("breakpoint_threshold", DEFAULT_BREAKPOINT_THRESHOLD)
        
        print("Configuration loaded. UI will update on next rerun.")


    # --- Processed Documents ---
    def store_processed_document(self, document_data: Dict[str, Any]):
        st.session_state.processed_documents.append(document_data)

    def get_all_processed_documents(self) -> List[Dict[str, Any]]:
        return st.session_state.processed_documents

    def clear_processed_documents(self):
        st.session_state.processed_documents = []
        self.index = None

    # --- Uploaded Files ---
    def store_uploaded_files(self, files: List[Any]):
        st.session_state.uploaded_files = files

    def get_uploaded_files(self) -> List[Any]:
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
    def index(self) -> Any:
        return st.session_state.get("index")

    @index.setter
    def index(self, value: Any):
        st.session_state.index = value

    # --- API Key Settings ---
    def get_api_key(self, provider: str = "openai") -> str:
        return st.session_state.api_keys_internal.get(provider, "")

    def update_api_key(self, key: str, provider: str = "openai"):
        st.session_state.api_keys_internal[provider] = key

    @property
    def api_key_valid(self) -> bool:
        return st.session_state.api_key_valid

    @api_key_valid.setter
    def api_key_valid(self, value: bool):
        st.session_state.api_key_valid = value

    # --- Model Settings ---
    @property
    def llm_model(self) -> str:
        return st.session_state.llm_model

    @llm_model.setter
    def llm_model(self, value: str):
        st.session_state.llm_model = value

    @property
    def embedding_model(self) -> str:
        return st.session_state.embedding_model

    @embedding_model.setter
    def embedding_model(self, value: str):
        st.session_state.embedding_model = value

    @property
    def llm_provider(self) -> str:
        return st.session_state.llm_provider_internal

    @llm_provider.setter
    def llm_provider(self, value: str):
        st.session_state.llm_provider_internal = value

    def get_model_settings(self) -> Dict[str, str]:
        """Returns a dictionary of current model settings."""
        return {
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "llm_provider": self.llm_provider
        }

    def update_model_settings(self, embedding_model: Optional[str] = None,
                              llm_model: Optional[str] = None,
                              llm_provider: Optional[str] = None):
        if embedding_model is not None:
            self.embedding_model = embedding_model
        if llm_model is not None:
            self.llm_model = llm_model
        if llm_provider is not None:
            self.llm_provider = llm_provider


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
        return st.session_state.chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        st.session_state.chunk_size = value

    @property
    def chunk_overlap(self) -> int:
        return st.session_state.chunk_overlap

    @chunk_overlap.setter
    def chunk_overlap(self, value: int):
        st.session_state.chunk_overlap = value

    # --- Semantic Splitter Options ---
    @property
    def buffer_size(self) -> int:
        return st.session_state.buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int):
        st.session_state.buffer_size = value

    @property
    def breakpoint_threshold(self) -> int:
        return st.session_state.breakpoint_threshold

    @breakpoint_threshold.setter
    def breakpoint_threshold(self, value: int):
        st.session_state.breakpoint_threshold = value

    def update_processing_options(self,
                                splitter_type: str,
                                vector_store_type: str,
                                chunk_size: Optional[int] = None,
                                chunk_overlap: Optional[int] = None,
                                buffer_size: Optional[int] = None,
                                breakpoint_threshold: Optional[int] = None):
        """Updates processing options. Specific splitter params are only updated if provided."""
        self.splitter_type = splitter_type
        self.vector_store_type = vector_store_type

        if splitter_type == "sentence":
            if chunk_size is not None: self.chunk_size = chunk_size
            if chunk_overlap is not None: self.chunk_overlap = chunk_overlap
        elif splitter_type == "semantic":
            if buffer_size is not None: self.buffer_size = buffer_size
            if breakpoint_threshold is not None: self.breakpoint_threshold = breakpoint_threshold
        else:
            st.warning(f"Unknown splitter type: {splitter_type} in update_processing_options")


    def get_processing_options(self) -> Dict[str, Any]:
        """Returns a dictionary of current processing options relevant to the active splitter."""
        options = {
            "splitter_type": self.splitter_type,
            "vector_store_type": self.vector_store_type,
        }
        if self.splitter_type == "sentence":
            options["chunk_size"] = self.chunk_size
            options["chunk_overlap"] = self.chunk_overlap
        elif self.splitter_type == "semantic":
            options["buffer_size"] = self.buffer_size
            options["breakpoint_threshold"] = self.breakpoint_threshold
        else:
            st.warning(f"Unknown splitter type: {self.splitter_type} in get_processing_options.")
            # Fallback to sentence defaults for the specific params
            options["chunk_size"] = DEFAULT_CHUNK_SIZE
            options["chunk_overlap"] = DEFAULT_CHUNK_OVERLAP
        return options

    # --- Chat History ---
    def get_session_chat_history(self) -> List[str]:
        return st.session_state.chat_history

    def update_session_chat_history(self, message: str):
        st.session_state.chat_history.append(message)

    def clear_session_chat_history(self):
        st.session_state.chat_history = []

    # --- Evaluation Run Management ---
    def add_evaluation_run(self, run_data: Dict[str, Any]):
        st.session_state.evaluation_runs.append(run_data)

    def get_all_evaluation_runs(self) -> List[Dict[str, Any]]:
        return st.session_state.get("evaluation_runs", [])

    def get_evaluation_run_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for run in st.session_state.get("evaluation_runs", []):
            if run.get("name") == name:
                return run
        return None

    def clear_all_evaluation_runs(self):
        st.session_state.evaluation_runs = []

    # --- Clear All Session Data ---
    def clear_all_session_data(self):
        """Clears most user-generated data from the session, resetting to a near-initial state."""
        self.clear_processed_documents()
        self.clear_uploaded_files()
        self.clear_qa_data()
        self.clear_session_chat_history()
        self.clear_all_evaluation_runs()

        # Reset settings to their initial defaults from config
        st.session_state.api_keys_internal = {"openai": ""}
        st.session_state.openai_key = ""
        st.session_state.api_key_valid = False
        
        # Reset configuration management
        st.session_state.loaded_config_data_state = None
        st.session_state.last_uploaded_config_id = None

        # Reset model settings (direct widget keys)
        st.session_state.embedding_model = DEFAULT_EMBEDDING_MODEL
        st.session_state.llm_model = DEFAULT_LLM_MODEL
        st.session_state.llm_provider_internal = DEFAULT_LLM_PROVIDER

        # Reset processing options (direct widget keys)
        st.session_state.splitter_type = DEFAULT_SPLITTER_TYPE
        st.session_state.vector_store_type = DEFAULT_VECTOR_STORE_TYPE
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP
        st.session_state.buffer_size = DEFAULT_BUFFER_SIZE
        st.session_state.breakpoint_threshold = DEFAULT_BREAKPOINT_THRESHOLD
