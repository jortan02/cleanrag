import streamlit as st
from utils.data_utils import initialize_session_state
from utils.api_config import (
    update_api_key,
    get_api_key,
    update_model_settings,
    get_model_settings,
    validate_openai_key
)

# Initialize session state
initialize_session_state()

def handle_api_key_change():
    """Handle API key changes and validation."""
    if st.session_state.openai_key:
        if validate_openai_key(st.session_state.openai_key):
            st.session_state.api_key_valid = True
            update_api_key("openai", st.session_state.openai_key)
        else:
            st.session_state.api_key_valid = False

def handle_model_change():
    """Handle model selection changes."""
    update_model_settings(
        embedding_model=st.session_state.embedding_model,
        llm_model=st.session_state.llm_model,
        llm_provider="openai"  # Always OpenAI now
    )

st.set_page_config(
    page_title="CleanRAG - Setup", page_icon="⚙️", layout="wide"
)

st.title("Setup ⚙️")

# API Keys Section
st.header("API Keys")

# OpenAI API Key with callback
st.text_input(
    "OpenAI API Key",
    value=get_api_key("openai"),
    type="password",
    help="Required for GPT models and OpenAI embeddings",
    key="openai_key",
    on_change=handle_api_key_change
)

# Show validation status
if "api_key_valid" in st.session_state:
    if st.session_state.api_key_valid:
        st.success("✅ API key is valid")
    else:
        st.error("❌ Invalid API key. Please check and try again.")
else:
    st.warning("⚠️ Please enter your OpenAI API key to continue")

# Only show model settings if API key is valid
if st.session_state.get("api_key_valid", False):
    # Model Settings Section
    st.header("Model Settings")

    # Get current settings
    current_settings = get_model_settings()

    # LLM Model Selection with callback
    st.selectbox(
        "OpenAI Model",
        options=[
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ],
        index=0 if current_settings["llm_model"] == "gpt-3.5-turbo" else 1,
        help="Select the OpenAI model to use",
        key="llm_model",
        on_change=handle_model_change
    )

    # Embedding Model Selection with callback
    st.selectbox(
        "Embedding Model",
        options=[
            "text-embedding-3-small",  # OpenAI's latest
            "text-embedding-3-large"   # OpenAI's largest
        ],
        index=0 if current_settings["embedding_model"] == "text-embedding-3-small" else 1,
        help="Select the OpenAI embedding model to use",
        key="embedding_model",
        on_change=handle_model_change
    )

    # Model Information
    st.header("Model Information")

    st.markdown("""
    ### Available Models

    #### LLM Models
    - **OpenAI**
      - GPT-3.5 Turbo: Fast, cost-effective, good for most tasks
      - GPT-4: More capable, better reasoning, higher cost
      - GPT-4 Turbo: Latest version with improved performance

    #### Embedding Models
    - **OpenAI**
      - text-embedding-3-small: Fast, good quality, 1536 dimensions
      - text-embedding-3-large: Best quality, 3072 dimensions
      - Runs on OpenAI's servers
      - Costs money per token

    ### Notes
    - OpenAI API key is required for all models
    - All processing happens on OpenAI's servers
    """)
else:
    st.info("Please enter a valid OpenAI API key to configure model settings.") 