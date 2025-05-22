from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
import streamlit as st
import openai

def update_api_key(provider: str, key: str):
    """Update API key for a specific provider."""
    st.session_state.api_keys[provider] = key


def get_api_key(provider: str) -> str:
    """Get API key for a specific provider."""
    return st.session_state.api_keys.get(provider, "")


def update_model_settings(
    embedding_model: str = None,
    llm_model: str = None,
    llm_provider: str = None
):
    """Update model settings. """
    if embedding_model:
        st.session_state.model_settings["embedding_model"] = embedding_model
        # Initialize embedding model
        if embedding_model.startswith("text-embedding"):
            # OpenAI embedding
            embedding_model = OpenAIEmbedding(
                model=embedding_model,
                api_key=get_api_key("openai")
            )
        else:
            raise Exception(f"Error: The embedding model \'{embedding_model}\' could not be found.")
        Settings.embed_model = embedding_model
    if llm_model:
        st.session_state.model_settings["llm_model"] = llm_model
        if llm_provider == "openai":
            llm = OpenAI(
                model=llm_model,
                api_key=get_api_key("openai")
            )
        else:
            raise Exception(f"Error: The LLM \'{llm_model}\' could not be found.")
        Settings.llm = llm
    if llm_provider:
        st.session_state.model_settings["llm_provider"] = llm_provider


def get_model_settings() -> dict:
    """Get current model settings."""
    return st.session_state.model_settings


def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key by making a test request."""
    if not api_key:
        return False
    
    try:
        openai.api_key = api_key
        # Make a minimal API call to validate the key
        openai.models.list()
        return True
    except Exception:
        return False


def is_app_configured() -> bool:
    """Check if the app is properly configured with valid API keys."""
    openai_key = get_api_key("openai")
    return bool(openai_key and validate_openai_key(openai_key))
