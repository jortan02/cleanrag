# --- Document Processing Settings ---
DEFAULT_CHUNK_SIZE = 500  # For sentence splitter: target size in tokens/characters
DEFAULT_CHUNK_OVERLAP = 50 # For sentence splitter: overlap in tokens/characters
DEFAULT_BUFFER_SIZE = 1   # For semantic splitter: number of sentences to buffer
DEFAULT_BREAKPOINT_THRESHOLD = 95 # For semantic splitter: percentile threshold for breakpoints

# --- Default Model and API Settings ---
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small" # Or your preferred default
DEFAULT_LLM_MODEL = "gpt-3.5-turbo" # Or your preferred default
DEFAULT_LLM_PROVIDER = "openai" # Currently only supporting OpenAI

# --- Default Processing Options ---
DEFAULT_SPLITTER_TYPE = "sentence" # Options: "sentence", "semantic"
DEFAULT_VECTOR_STORE_TYPE = "simple" # Options: "simple", "faiss"

# --- File Type Settings ---
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx"]
SUPPORTED_QA_FILE_TYPES = ["csv"]
