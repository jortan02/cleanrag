from typing import List, Any
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.settings import Settings

# Use TYPE_CHECKING for SessionManager to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .session_manager import SessionManager # Assuming SessionManager is in utils.session_manager
    from llama_index.core.indices.base import BaseIndex # For type hinting index

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

def get_chat_engine(sm: 'SessionManager') -> BaseChatEngine | None:
    """
    Retrieves or creates a chat engine based on the index in SessionManager.
    Ensures LlamaIndex global settings are configured.
    """
    index: 'BaseIndex' | None = sm.index
    if not index:
        print("Error: Index not found in SessionManager. Cannot create chat engine.")
        return None

    # It's good practice to ensure LlamaIndex settings are current before creating engine
    # This could be called here, or rely on Home.txt callbacks to have done it.
    # For robustness, consider calling it, but be mindful of circular dependencies.
    # from .api_config import configure_llama_index_settings
    # if not Settings.llm or not Settings.embed_model: # Basic check if settings might be missing
    #     print("Re-configuring LlamaIndex settings before creating chat engine.")
    #     configure_llama_index_settings(sm)

    if not Settings.llm:
         print("LLM not configured in LlamaIndex Settings. Cannot create chat engine.")
         # Potentially raise an error or return a message to be displayed in UI
         return None

    try:
        # You can customize chat_mode, similarity_top_k, etc.
        chat_engine = index.as_chat_engine(
            chat_mode="condense_question", # Example, choose appropriate
            verbose=True # Useful for debugging
        )
        return chat_engine
    except Exception as e:
        print(f"Error creating chat engine: {e}")
        return None


def generate_response(query: str, sm: 'SessionManager') -> str:
    """
    Generate a response from the model using the index and chat history
    managed by SessionManager.
    """
    chat_engine = get_chat_engine(sm)
    if not chat_engine:
        return "Error: Chat engine could not be initialized. Please check configuration and processed documents."

    # Fetch and convert chat history from SessionManager
    # The user's current 'query' is the latest message, not yet in full history for this turn.
    raw_chat_history: List[str] = sm.get_session_chat_history() # This history is *before* the current user query
    
    # The raw_chat_history currently INCLUDES the user's latest query because Home.txt adds it
    # to sm.update_session_chat_history(user_input) BEFORE calling generate_response.
    # So, the history for the chat engine should be raw_chat_history[:-1] if user_input was just added.
    # Let's adjust Home.txt logic: user_input added to sm AFTER response.
    # For now, assuming raw_chat_history is what LlamaIndex needs (all prior messages).
    
    converted_chat_history = _convert_str_history_to_chatmessages(raw_chat_history)

    try:
        response = chat_engine.chat(query, chat_history=converted_chat_history)
        return response.response
    except Exception as e:
        print(f"Error during chat generation: {e}")
        # This error should ideally be caught and displayed gracefully in Home.txt
        return f"Sorry, an error occurred while generating the response: {str(e)}"