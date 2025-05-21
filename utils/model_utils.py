from typing import List, Dict, Any
import torch

torch.classes.__path__ = []
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import config
