from typing import List, Dict, Any
from llama_index.llms.openai import OpenAI
# from utils.data_utils
import streamlit as st

def generate_response(query: str, chat_history: List[Dict[str, Any]] = []) -> str:
    """Generate a response from the model."""
    index = st.session_state.index
    chat_engine = index.as_chat_engine()
    
    response = chat_engine.chat(query, chat_history=chat_history)
    return response.response
