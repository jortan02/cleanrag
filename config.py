import torch

# Application settings
LOCAL_MODE = False  # Set to True for local development
USE_GPU = torch.cuda.is_available() if LOCAL_MODE else False
DEBUG = False  # Set to True for debug mode

# Document processing settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# File type settings
SUPPORTED_FILE_TYPES = ["txt", "pdf", "docx"]
SUPPORTED_QA_FILE_TYPES = ["csv"]
