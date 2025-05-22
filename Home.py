import streamlit as st
import asyncio
import pandas as pd
from llama_index.core import Document
from llama_index.core.settings import Settings

from utils.session_manager import SessionManager
from utils.api_config import (
    validate_openai_key,
    configure_llama_index_settings,
    is_app_configured,
)
from utils.document_utils import (
    extract_text_from_file,
    process_document,
    get_chunk_statistics,
    create_chunk_dataframe,
    create_index_from_documents,
    # VectorStoreType, # Not directly used here, but create_index_from_documents might
)
from utils.model_utils import generate_response
from utils.evaluation_utils import run_ragas_evaluation
from config import SUPPORTED_FILE_TYPES, COL_QUESTION, COL_ANSWER_GROUND_TRUTH, COL_CONTEXTS_GROUND_TRUTH

# --- Initialize Session Manager ---
# This MUST be one of the first Streamlit commands.
# The SessionManager constructor handles initializing all necessary session state keys.
sm = SessionManager()

# --- Setup asyncio event loop (if truly necessary for other non-Streamlit parts) ---
# For Streamlit itself, this is usually not required.
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# --- Page Config ---
st.set_page_config(
    page_title="CleanRAG",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initial LlamaIndex Configuration Attempt ---
# If an API key was already validated and stored from a previous session,
# configure LlamaIndex settings on load.
if sm.api_key_valid and sm.get_api_key("openai"):
    # Wrap in a try-except as API key might have become invalid since last validation
    try:
        configure_llama_index_settings(sm)
        print("LlamaIndex settings configured on page load from existing valid API key.")
    except Exception as e:
        print(f"Error configuring LlamaIndex on page load: {e}")
        # Optionally, mark API key as invalid in session manager if config fails
        # sm.api_key_valid = False # This might be too aggressive

# --- Callback Functions ---
def handle_api_key_change():
    """Handle API key changes and validate."""
    api_key_from_input = st.session_state.openai_key # Value from the text_input widget
    if api_key_from_input:
        is_valid = validate_openai_key(api_key_from_input)
        sm.api_key_valid = is_valid  # Update SessionManager's flag
        if is_valid:
            sm.update_api_key(api_key_from_input) # Store the valid key in SessionManager
        else:
            sm.update_api_key("") # Clear invalid key in SessionManager if desired
    else:
        sm.api_key_valid = False
        sm.update_api_key("") # Clear key if input is empty
    
    # (Re)configure LlamaIndex global settings after any key change or validation attempt
    try:
        configure_llama_index_settings(sm)
    except Exception as e:
        st.error(f"Failed to configure model settings with new API key: {e}")


def handle_embedding_model_change():
    """Handle embedding model selection changes."""
    if sm.api_key_valid:
        selected_embedding_model = st.session_state.embedding_model # Value from selectbox widget
        sm.embedding_model = selected_embedding_model # Update SessionManager
        try:
            configure_llama_index_settings(sm) # Re-apply global LlamaIndex settings
        except Exception as e:
            st.error(f"Failed to configure embedding model: {e}")
    else:
        st.warning("Please ensure your API key is valid before changing model settings.")

def handle_llm_change():
    """Handle LLM model selection changes."""
    if sm.api_key_valid:
        selected_llm_model = st.session_state.llm_model # Value from selectbox widget
        sm.llm_model = selected_llm_model # Update SessionManager
        # Assuming OpenAI provider for this UI element. SessionManager defaults should handle this.
        # sm.update_model_settings(llm_provider="openai") # If provider can change via UI later
        try:
            configure_llama_index_settings(sm) # Re-apply global LlamaIndex settings
        except Exception as e:
            st.error(f"Failed to configure LLM: {e}")
    else:
        st.warning("Please ensure your API key is valid before changing model settings.")

def handle_splitter_change():
    """Handle splitter type changes.
    The widget's `key` directly updates `st.session_state.splitter_type`.
    SessionManager's `splitter_type` property reflects this.
    This callback ensures processing options are consistently applied based on the NEW type.
    """
    new_splitter = sm.splitter_type # This will be the new value from the selectbox
    current_vector_store = sm.vector_store_type

    if new_splitter == "sentence":
        param1 = sm.chunk_size # Use existing/default sentence chunk_size
        param2 = sm.chunk_overlap # Use existing/default sentence chunk_overlap
    else:  # semantic
        param1 = sm.buffer_size # Use existing/default semantic buffer_size
        param2 = sm.breakpoint_threshold # Use existing/default semantic breakpoint_threshold
    
    sm.update_processing_options(param1, param2, new_splitter, current_vector_store)
    # UI will update on rerun due to widget change.

def handle_processing_options_change():
    """Handle processing options changes from sliders or vector_store_type selectbox."""
    active_splitter = sm.splitter_type # Current active splitter
    
    # Values are from the st.session_state.widget_key of the sliders/selectbox
    if active_splitter == "sentence":
        param1 = st.session_state.chunk_size
        param2 = st.session_state.chunk_overlap
    else:  # semantic
        param1 = st.session_state.buffer_size
        param2 = st.session_state.breakpoint_threshold
            
    new_vector_store_type = st.session_state.vector_store_type

    sm.update_processing_options(param1, param2, active_splitter, new_vector_store_type)
    # UI will update on rerun.

# --- Navigation ---
st.sidebar.markdown(
    """
### 📚 Navigation

- 🏠 [**Home**](#home)
- ⚙️ [**Setup**](#setup)
- 📁 [**Upload**](#upload)
- 💬 [**Chat**](#chat)
- 📊 [**Evaluate**](#evaluate)
""",
    unsafe_allow_html=True,
)

# --- Home Section ---
st.header("Welcome to CleanRAG 🧹", anchor="home")
st.markdown(
    """
### Your RAG Pipeline Optimization Tool

CleanRAG helps you analyze, optimize, and debug your Retrieval-Augmented Generation (RAG) pipeline.

#### Quick Start:
1. **Setup**: Configure your OpenAI API key and model settings.
2. **Upload**: Add your documents and process them.
3. **Chat**: Interact with your documents.
4. **Evaluate**: Analyze your RAG pipeline's performance.

#### Current Status:
"""
)

# Show current status
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        "API Key",
        "✅ Configured" if sm.api_key_valid else "❌ Not Configured",
    )
with col2:
    docs = sm.get_all_processed_documents()
    st.metric(
        "Documents", f"{len(docs)} Processed" if docs else "No Documents",
    )
with col3:
    st.metric(
        "Index",
        "✅ Ready" if sm.index is not None else "❌ Not Created",
    )
with col4:
    qa_data_df = sm.get_qa_data()
    st.metric(
        "QA Pairs",
        f"{len(qa_data_df)} Loaded" if qa_data_df is not None else "No QA Data",
    )

# --- Setup Section ---
st.header("Setup ⚙️", anchor="setup")

# API Keys Section
st.subheader("API Key") # Changed to subheader for better hierarchy

# OpenAI API Key with callback
st.text_input( # No need to assign to variable if only used by callback via st.session_state
    "OpenAI API Key",
    value=sm.get_api_key("openai"), # Get initial value from SessionManager
    type="password",
    help="Required for GPT models and OpenAI embeddings.",
    key="openai_key", # This key is used by the callback
    on_change=handle_api_key_change,
)

# Show validation status directly after input
if "openai_key" in st.session_state and st.session_state.openai_key: # Check if key has been entered
    if sm.api_key_valid:
        # Success message handled in callback or here for persistence
        st.success("✅ API key is valid.")
    else:
        # Error message handled in callback or here
        st.error("❌ Invalid API key. Please check and try again.")
elif not sm.get_api_key("openai"): # If no key is stored in sm and input is also empty
    st.warning("⚠️ Please enter your OpenAI API key to continue.")


# Model Settings Section
st.subheader("Model Settings") # Changed to subheader

# Get current settings for selectbox indices
current_llm_model = sm.llm_model
current_embedding_model = sm.embedding_model
app_is_configured = is_app_configured(sm)

# LLM Model Selection with callback
st.selectbox(
    "OpenAI Model",
    options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"],
    index=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"].index(current_llm_model),
    help="Select the OpenAI model to use.",
    key="llm_model", # Used by callback
    on_change=handle_llm_change,
    disabled=not app_is_configured,
)

# Embedding Model Selection with callback
st.selectbox(
    "Embedding Model",
    options=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
    index=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"].index(current_embedding_model),
    help="Select the OpenAI embedding model to use.",
    key="embedding_model", # Used by callback
    on_change=handle_embedding_model_change,
    disabled=not app_is_configured,
)

# --- Upload Section ---
st.header("Upload & Process 📁", anchor="upload")

# File upload section
st.subheader("Upload Documents")

# File uploader
uploaded_files_widget = st.file_uploader(
    "Choose your documents",
    type=SUPPORTED_FILE_TYPES,
    accept_multiple_files=True,
    key="file_uploader_widget", # Use a distinct key for the widget
)
if uploaded_files_widget:
    sm.store_uploaded_files(uploaded_files_widget) # Store them in session manager

# Processing options
st.subheader("Processing Options")

# Splitter type selection with callback
st.selectbox(
    "Splitter Type",
    options=["sentence", "semantic"],
    index=["sentence", "semantic"].index(sm.splitter_type),
    format_func=lambda x: "Sentence-based" if x == "sentence" else "Semantic",
    help="Choose how to split documents.",
    key="splitter_type", # Used by callback and sm property
    on_change=handle_splitter_change,
    disabled=not app_is_configured,
)

# Show different controls based on splitter type
active_splitter = sm.splitter_type # Get current active splitter from SessionManager

if active_splitter == "sentence":
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Chunk Size", min_value=100, max_value=2000,
            value=sm.chunk_size, step=100,
            help="Target size for each chunk in tokens.",
            key="chunk_size", # Used by callback
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )
    with col2:
        st.slider(
            "Chunk Overlap", min_value=0, max_value=200,
            value=sm.chunk_overlap, step=10,
            help="Number of tokens to overlap between chunks.",
            key="chunk_overlap", # Used by callback
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )
else:  # semantic
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Buffer Size", min_value=1, max_value=5,
            value=sm.buffer_size, step=1,
            help="Number of sentences to buffer. Higher values may result in more natural splits.",
            key="buffer_size", # Used by callback
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )
    with col2:
        st.slider(
            "Breakpoint Threshold", min_value=50, max_value=99,
            value=sm.breakpoint_threshold, step=1,
            help="Percentile threshold for determining breakpoints. Higher values result in fewer splits.",
            key="breakpoint_threshold", # Used by callback
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )

# Vector store options with callback
st.selectbox(
    "Vector Store Type",
    options=["simple", "faiss"],
    index=["simple", "faiss"].index(sm.vector_store_type),
    format_func=lambda x: "Simple" if x == "simple" else "FAISS",
    help="Choose the vector store type. FAISS provides better performance for larger datasets.",
    key="vector_store_type", # Used by callback
    on_change=handle_processing_options_change,
    disabled=not app_is_configured,
)

# Process button
stored_files_for_processing = sm.get_uploaded_files() # Get latest from SM
if st.button("Process Documents", type="primary", disabled=not app_is_configured or not stored_files_for_processing):
    sm.clear_processed_documents() # Clears documents and index in SM
    if stored_files_for_processing: # Check again due to button state
        progress_bar = st.progress(0, "Initializing document processing...")
        status_text = st.empty()
        total_files = len(stored_files_for_processing)

        for i, file_data in enumerate(stored_files_for_processing):
            actual_filename = file_data.name if hasattr(file_data, 'name') else f"File {i+1}"
            status_text.text(f"Processing {actual_filename} ({i+1}/{total_files})...")
            try:
                text = extract_text_from_file(file_data)
                doc = Document(text=text, metadata={"filename": actual_filename})

                current_splitter_type = sm.splitter_type
                if current_splitter_type == "sentence":
                    kwargs = {"chunk_size": sm.chunk_size, "chunk_overlap": sm.chunk_overlap}
                else:  # semantic
                    kwargs = {"buffer_size": sm.buffer_size, "breakpoint_percentile_threshold": sm.breakpoint_threshold}
                
                chunks = process_document(doc, splitter_type=current_splitter_type, **kwargs) # Pass LlamaIndex Document to process_document
                
                stats = get_chunk_statistics(chunks) # Assumes chunks are LlamaIndex NodeWithScore or similar
                chunk_df = create_chunk_dataframe(chunks)

                document_data = {
                    "name": actual_filename, "text": text, "chunks": chunks,
                    "stats": stats, "chunk_df": chunk_df, "document": doc,
                }
                sm.store_processed_document(document_data)
                progress_bar.progress((i + 1) / total_files, f"Processed {actual_filename}")

            except Exception as e:
                st.error(f"Error processing {actual_filename}: {str(e)}")
        
        status_text.text("Creating index from processed documents...")
        try:
            all_docs_for_index = [data["document"] for data in sm.get_all_processed_documents()]
            if all_docs_for_index:
                # Ensure LlamaIndex settings are configured before creating index
                configure_llama_index_settings(sm)
                if not Settings.embed_model: # Check if embed_model was successfully configured
                     st.error("Embedding model not configured. Cannot create index. Please check API key and model settings.")
                else:
                    index = create_index_from_documents(
                        all_docs_for_index, 
                        vector_store_type=sm.vector_store_type
                        # embed_model will be taken from Settings.embed_model
                    )
                    sm.index = index # Store the index in SessionManager
                    status_text.text("") # Clear status text
                    progress_bar.empty() # Clear progress bar
                    st.success("Documents processed and indexed successfully!")
                    st.rerun() # Rerun to update UI, especially "Index Ready" status
            else:
                st.warning("No documents were successfully processed to create an index.")

        except Exception as e:
            st.error(f"Error creating index: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    else:
        st.warning("Please upload at least one document.")

processed_docs_display = sm.get_all_processed_documents()
if processed_docs_display:
    st.subheader("Processed Document Details")
    for i, doc_data_item in enumerate(processed_docs_display):
        is_last = i == len(processed_docs_display) - 1
        with st.expander(f"📄 {doc_data_item['name']}", expanded=is_last):
            st.markdown("##### Document Statistics") # Smaller heading
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Chunks", doc_data_item["stats"]["total_chunks"])
            with col2: st.metric("Avg Chunk Size", f"{doc_data_item['stats']['avg_chunk_size_chars']:.1f} characters") # Assuming stats are char based
            with col3: st.metric("Total Words", doc_data_item["stats"]["total_words"]) # Or characters if preferred

            st.markdown("##### Chunk Details")
            st.dataframe(doc_data_item["chunk_df"], use_container_width=True, height=200)

# QA File upload section
st.subheader("Upload QA Test Set for Evaluation")
st.markdown(
    f"""
    Upload a CSV file. Required columns:
    - `{COL_QUESTION}`: The question to ask the RAG system.
    - `{COL_ANSWER_GROUND_TRUTH}`: Your ground truth or expected answer for the question.

    Optional column for more detailed context evaluation:
    - `{COL_CONTEXTS_GROUND_TRUTH}`: A list of ideal context strings that should support the answer.
      Format this cell as a string representation of a list (e.g., `["ideal context 1 text", "another ideal context"]`)
      or a single string if there's only one ideal context.
      Providing this enables more effective 'Context Recall' evaluation.
    """
)

qa_file_widget = st.file_uploader(
    "Choose your QA test set (CSV)",
    type=["csv"],
    key="qa_file_uploader_widget",
    disabled=not app_is_configured,
)

if qa_file_widget is not None:
    try:
        new_qa_data = pd.read_csv(qa_file_widget)
        # Check for required columns
        if COL_QUESTION not in new_qa_data.columns or COL_ANSWER_GROUND_TRUTH not in new_qa_data.columns:
            st.error(f"QA file must contain '{COL_QUESTION}' and '{COL_ANSWER_GROUND_TRUTH}' columns.")
            # Optionally clear previously loaded data if new upload is invalid
            # sm.store_qa_data(None) 
        else:
            prev_qa_data = sm.get_qa_data()
            if prev_qa_data is None or not new_qa_data.equals(prev_qa_data):
                sm.store_qa_data(new_qa_data)
                st.success(f"QA test set '{qa_file_widget.name}' loaded successfully.")
                if COL_CONTEXTS_GROUND_TRUTH in new_qa_data.columns and new_qa_data[COL_CONTEXTS_GROUND_TRUTH].notna().any():
                    st.info(f"Optional '{COL_CONTEXTS_GROUND_TRUTH}' column detected and will be used for evaluation.")
                st.rerun()
    except Exception as e:
        st.error(f"Error reading or processing QA file: {e}")
        sm.store_qa_data(None) # Clear QA data in session manager on error

# Display QA data if available
current_qa_data_df = sm.get_qa_data()
if current_qa_data_df is not None:
    st.markdown("##### Loaded QA Test Set (Preview)")
    st.dataframe(current_qa_data_df.head(), use_container_width=True)


# --- Chat Section ---
st.header("Chat 💬", anchor="chat")

with st.container(border=True):
    chat_history_display = sm.get_session_chat_history()
    for i, message_content in enumerate(chat_history_display):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.write(message_content)

    user_query = st.chat_input(
        placeholder="Ask a question about your documents...",
        disabled=not app_is_configured or not sm.index, # Disable if no API key or no index
    )

    if user_query:
        # Add user message to SM history *before* calling generate.
        # `generate_response` will then use this full history. (Adjust if model_utils expects otherwise)
        # sm.update_session_chat_history(user_query) # This makes generate_response's history param complex
                                                # Better: pass query, let generate_response use sm's history *before* this query
        
        # Ensure LlamaIndex settings are configured before chat
        try:
            configure_llama_index_settings(sm)
            if not Settings.llm: # Check if LLM was successfully configured
                st.error("LLM not configured. Cannot chat. Please check API key and model settings.")
            else:
                assistant_response = generate_response(user_query, sm=sm)
                sm.update_session_chat_history(user_query) # Store user query
                sm.update_session_chat_history(assistant_response) # Store assistant response
                st.rerun()
        except Exception as e:
            st.error(f"Error during chat: {e}")


if st.button("Clear Chat History", use_container_width=True, disabled=not chat_history_display):
    sm.clear_session_chat_history()
    st.rerun()

# --- Evaluate Section ---
st.header("Evaluate RAG Pipeline 📊", anchor="evaluate")

current_qa_data_df = sm.get_qa_data()
app_is_configured = is_app_configured(sm)

if current_qa_data_df is not None and not current_qa_data_df.empty and sm.index is not None:
    if st.button("Run Evaluation with Ragas", type="primary", disabled=not app_is_configured):
        with st.spinner("Generating RAG outputs and running Ragas evaluation... This may take a while and consume API credits."):
            try:
                # 1. Configure LlamaIndex Settings for *your* RAG pipeline
                configure_llama_index_settings(sm) 
                if not Settings.llm or not Settings.embed_model:
                     st.error("LlamaIndex LLM or Embedding Model not configured. Cannot run evaluation.")
                else:
                    # 2. Ensure OPENAI_API_KEY is in env for Ragas's own evaluators
                    import os
                    openai_api_key = sm.get_api_key("openai")
                    if openai_api_key and not os.getenv("OPENAI_API_KEY"):
                        # Only set it if it's not already there to avoid overriding other settings
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                        print("Temporarily set OPENAI_API_KEY environment variable from SessionManager for Ragas.")
                    elif not openai_api_key and not os.getenv("OPENAI_API_KEY"):
                        st.warning("OpenAI API key not found in SessionManager or environment. Ragas evaluation might fail or use cached results if applicable.")

                    # 3. Run the evaluation
                    eval_results_df = run_ragas_evaluation(sm) # This calls the updated util function
                    
                    st.subheader("Ragas Evaluation Results")
                    
                    if 'ragas_score' in eval_results_df.columns:
                         st.metric("Overall Ragas Score (if provided by Ragas version)", f"{eval_results_df['ragas_score'].mean():.3f}")
                    
                    # Display individual metrics that were run
                    # These are the Ragas metric names by default
                    possible_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
                    
                    cols = st.columns(len([m for m in possible_metrics if m in eval_results_df.columns]))
                    col_idx = 0
                    for metric_name in possible_metrics:
                        if metric_name in eval_results_df.columns:
                            # Calculate mean, handling potential non-numeric or all-NaN cases
                            metric_series = pd.to_numeric(eval_results_df[metric_name], errors='coerce')
                            if not metric_series.empty and metric_series.notna().any():
                                mean_score = metric_series.mean()
                                cols[col_idx].metric(f"{metric_name.replace('_', ' ').title()}", f"{mean_score:.3f}")
                            else:
                                cols[col_idx].metric(f"{metric_name.replace('_', ' ').title()}", "N/A")
                            col_idx += 1
                    
                    st.dataframe(eval_results_df, use_container_width=True)

                    # Inform user about context_recall if GT contexts were not used
                    if COL_CONTEXTS_GROUND_TRUTH not in current_qa_data_df.columns or not current_qa_data_df[COL_CONTEXTS_GROUND_TRUTH].notna().any():
                        if "context_recall" not in eval_results_df.columns: # Check if it wasn't even run
                            st.info(f"Note: '{COL_CONTEXTS_GROUND_TRUTH}' column was not found or was empty in your QA data, so 'Context Recall' based on this specific input was not performed by Ragas or might reflect a different calculation.")
                        elif "context_recall" in eval_results_df.columns: # It ran, but inform about interpretation
                             st.info(f"Note: 'Context Recall' was computed. Without user-provided '{COL_CONTEXTS_GROUND_TRUTH}', its score reflects Ragas's assessment based on other available data (like the ground truth answer).")

            except Exception as e:
                st.error(f"An error occurred during the Ragas evaluation process: {e}")
                import traceback
                st.error(traceback.format_exc())

elif not app_is_configured:
    st.warning("RAG evaluation requires a valid API key and configured models in Setup.")
elif current_qa_data_df is None or current_qa_data_df.empty:
    st.warning("Please upload a QA test set (CSV with 'question' and 'answer' columns) to run evaluation.")
elif sm.index is None:
    st.warning("Please upload and process documents to create an index before running evaluation.")