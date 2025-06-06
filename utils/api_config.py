from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from openai import OpenAI as OpenAIClient

# Avoid circular imports if SessionManager is in another file
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .session_manager import SessionManager

def configure_llama_index_settings(sm: 'SessionManager'):
    """
    Configures global LlamaIndex settings (Settings.llm and Settings.embed_model)
    based on the current settings stored in SessionManager.

    This function should be called after API keys or model selections are confirmed
    and updated in the SessionManager.
    """
    api_key = sm.get_api_key("openai") # Get from SessionManager
    model_settings = sm.get_model_settings() # Get from SessionManager

    embedding_model_name = model_settings.get("embedding_model")
    llm_model_name = model_settings.get("llm_model")
    llm_provider = model_settings.get("llm_provider")

    # Only configure if the API key is considered valid by the SessionManager
    if not sm.api_key_valid or not api_key:
        # Clear existing LlamaIndex settings if API key is not valid or missing
        # to prevent using stale configurations.
        Settings.llm = None
        Settings.embed_model = None
        print("API key not valid or missing. LlamaIndex global settings cleared.")
        return

    # Configure Embedding Model
    if embedding_model_name:
        if embedding_model_name.startswith("text-embedding"):  # OpenAI embedding
            try:
                embed_model = OpenAIEmbedding(
                    model=embedding_model_name,
                    api_key=api_key
                )
                Settings.embed_model = embed_model
                print(f"LlamaIndex Settings.embed_model configured to: {embedding_model_name}")
            except Exception as e:
                Settings.embed_model = None # Clear on failure
                print(f"Error configuring LlamaIndex OpenAIEmbedding with {embedding_model_name}: {e}")
                # Error should be surfaced to user in Home.py
        else:
            Settings.embed_model = None
            print(f"Unsupported embedding model for LlamaIndex settings: {embedding_model_name}")
    else:
        Settings.embed_model = None # No embedding model specified

    # Configure LLM
    if llm_model_name and llm_provider:
        if llm_provider == "openai":
            try:
                llm = LlamaOpenAI(
                    model=llm_model_name,
                    api_key=api_key
                )
                Settings.llm = llm
                print(f"LlamaIndex Settings.llm configured to: {llm_model_name}")
            except Exception as e:
                Settings.llm = None # Clear on failure
                print(f"Error configuring LlamaIndex OpenAI LLM with {llm_model_name}: {e}")
                # Error should be surfaced to user in Home.py
        else:
            Settings.llm = None
            print(f"Unsupported LLM provider for LlamaIndex settings: {llm_provider}")
    else:
        Settings.llm = None # No LLM specified

def validate_openai_key(api_key: str) -> bool:
    """Validate OpenAI API key by making a test request."""
    if not api_key:
        return False
    try:
        client = OpenAIClient(api_key=api_key)
        client.models.list()  # Make a minimal API call to validate the key
        return True
    except Exception as e:
        print(f"OpenAI API key validation failed: {e}") # Log for debugging
        return False

def is_app_configured(sm: 'SessionManager') -> bool:
    """
    Check if the app is properly configured with a valid API key.
    Relies on SessionManager for the API key validity status and presence.
    """
    # sm.api_key_valid is the primary source of truth, set after validation.
    # Also check if the key string itself is present.
    return sm.api_key_valid and bool(sm.get_api_key("openai"))