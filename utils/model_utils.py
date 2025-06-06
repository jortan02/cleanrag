from typing import List, Any
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.settings import Settings
from .api_config import configure_llama_index_settings

# Use TYPE_CHECKING for SessionManager to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session_manager import SessionManager
    from llama_index.core.indices.base import BaseIndex
    from .api_config import configure_llama_index_settings


def _convert_str_history_to_chatmessages(str_history: List[str]) -> List[ChatMessage]:
    """
    Converts a flat list of strings (alternating user/assistant) from session state
    to a list of LlamaIndex ChatMessage objects.
    """
    messages: List[ChatMessage] = []
    for i, msg_content in enumerate(str_history):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        messages.append(ChatMessage(role=role, content=msg_content))
    return messages


def get_chat_engine(sm: "SessionManager") -> BaseChatEngine | None:
    """
    Retrieves or creates a chat engine based on the index in SessionManager.
    Ensures LlamaIndex global settings are configured.
    """
    index: "BaseIndex" | None = sm.index
    if not index:
        print("Error: Index not found in SessionManager. Cannot create chat engine.")
        return None

    # Basic check if settings might be missing
    if not Settings.llm or not Settings.embed_model:
        print("Re-configuring LlamaIndex settings before creating chat engine.")
        configure_llama_index_settings(sm)

    try:
        chat_engine = index.as_chat_engine(verbose=True)
        return chat_engine
    except Exception as e:
        print(f"Error creating chat engine: {e}")
        return None


def generate_response(query: str, sm: "SessionManager") -> str:
    """
    Generate a response from the model using the index and chat history
    managed by SessionManager.
    """
    chat_engine = get_chat_engine(sm)
    if not chat_engine:
        return "Error: Chat engine could not be initialized. Please check configuration and processed documents."

    # History does not include the user's latest query
    raw_chat_history: List[str] = sm.get_session_chat_history()
    converted_chat_history = _convert_str_history_to_chatmessages(raw_chat_history)

    try:
        response = chat_engine.chat(query, chat_history=converted_chat_history)
        return response.response
    except Exception as e:
        print(f"Error during chat generation: {e}")
        return f"Sorry, an error occurred while generating the response: {str(e)}"
