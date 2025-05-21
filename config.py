import os
from dotenv import load_dotenv
import torch

torch.classes.__path__ = []

# Load environment variables from .env file
load_dotenv()

# Application settings
LOCAL_MODE = os.getenv("LOCAL_MODE", "False").lower() == "true"
# Automatically detect CUDA availability
USE_GPU = torch.cuda.is_available() if LOCAL_MODE else False

# Model settings
if LOCAL_MODE:
    # Local model settings
    # Embedding model settings
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DEVICE = "cuda" if USE_GPU else "cpu"

    # LLM settings
    LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-v0.1")
    LLM_DEVICE = "cuda" if USE_GPU else "cpu"
    LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "4096"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
else:
    # Hosted model settings
    API_KEY = os.getenv("API_KEY", "")
    API_ENDPOINT = os.getenv("API_ENDPOINT", "")

# Application settings
APP_NAME = "My Streamlit App"
APP_ICON = "✨"
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
