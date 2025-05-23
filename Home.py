import io
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
)
from utils.model_utils import generate_response
from utils.evaluation_utils import prepare_and_run_ragas_evaluation
from config import (
    SUPPORTED_FILE_TYPES,
    COL_QUESTION,
    COL_ANSWER_GROUND_TRUTH,
    COL_CONTEXTS_GROUND_TRUTH,
)
import json
import base64
from datetime import datetime
import os
import traceback

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
        print(
            "LlamaIndex settings configured on page load from existing valid API key."
        )
    except Exception as e:
        print(f"Error configuring LlamaIndex on page load: {e}")
        # Optionally, mark API key as invalid in session manager if config fails
        # sm.api_key_valid = False # This might be too aggressive


# --- Callback Functions ---
def handle_api_key_change():
    """Handle API key changes and validate."""
    api_key_from_input = st.session_state.openai_key  # Value from the text_input widget
    if api_key_from_input:
        is_valid = validate_openai_key(api_key_from_input)
        sm.api_key_valid = is_valid  # Update SessionManager's flag
        if is_valid:
            sm.update_api_key(
                api_key_from_input
            )  # Store the valid key in SessionManager
        else:
            sm.update_api_key("")  # Clear invalid key in SessionManager if desired
    else:
        sm.api_key_valid = False
        sm.update_api_key("")  # Clear key if input is empty

    # (Re)configure LlamaIndex global settings after any key change or validation attempt
    try:
        configure_llama_index_settings(sm)
    except Exception as e:
        st.error(f"Failed to configure model settings with new API key: {e}")


def handle_embedding_model_change():
    """Handle embedding model selection changes."""
    if sm.api_key_valid:
        selected_embedding_model = (
            st.session_state.embedding_model
        )  # Value from selectbox widget
        sm.embedding_model = selected_embedding_model  # Update SessionManager
        try:
            configure_llama_index_settings(sm)  # Re-apply global LlamaIndex settings
        except Exception as e:
            st.error(f"Failed to configure embedding model: {e}")
    else:
        st.warning(
            "Please ensure your API key is valid before changing model settings."
        )


def handle_llm_change():
    """Handle LLM model selection changes."""
    if sm.api_key_valid:
        selected_llm_model = st.session_state.llm_model  # Value from selectbox widget
        sm.llm_model = selected_llm_model  # Update SessionManager
        # Assuming OpenAI provider for this UI element. SessionManager defaults should handle this.
        # sm.update_model_settings(llm_provider="openai") # If provider can change via UI later
        try:
            configure_llama_index_settings(sm)  # Re-apply global LlamaIndex settings
        except Exception as e:
            st.error(f"Failed to configure LLM: {e}")
    else:
        st.warning(
            "Please ensure your API key is valid before changing model settings."
        )


def handle_splitter_change():
    """Handle splitter type changes.
    The widget's `key` directly updates `st.session_state.splitter_type`.
    SessionManager's `splitter_type` property reflects this.
    This callback ensures processing options are consistently applied based on the NEW type.
    """
    new_splitter = sm.splitter_type  # This will be the new value from the selectbox
    current_vector_store = sm.vector_store_type

    if new_splitter == "sentence":
        param1 = sm.chunk_size  # Use existing/default sentence chunk_size
        param2 = sm.chunk_overlap  # Use existing/default sentence chunk_overlap
    else:  # semantic
        param1 = sm.buffer_size  # Use existing/default semantic buffer_size
        param2 = (
            sm.breakpoint_threshold
        )  # Use existing/default semantic breakpoint_threshold

    sm.update_processing_options(param1, param2, new_splitter, current_vector_store)
    # UI will update on rerun due to widget change.


def handle_processing_options_change():
    """Handle processing options changes from sliders or vector_store_type selectbox."""
    active_splitter = sm.splitter_type  # Current active splitter

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
        "Documents",
        f"{len(docs)} Processed" if docs else "No Documents",
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
st.subheader("API Key")  # Changed to subheader for better hierarchy

# OpenAI API Key with callback
st.text_input(  # No need to assign to variable if only used by callback via st.session_state
    "OpenAI API Key",
    value=sm.get_api_key("openai"),  # Get initial value from SessionManager
    type="password",
    help="Required for GPT models and OpenAI embeddings.",
    key="openai_key",  # This key is used by the callback
    on_change=handle_api_key_change,
)

# Show validation status directly after input
if (
    "openai_key" in st.session_state and st.session_state.openai_key
):  # Check if key has been entered
    if sm.api_key_valid:
        # Success message handled in callback or here for persistence
        st.success("✅ API key is valid.")
    else:
        # Error message handled in callback or here
        st.error("❌ Invalid API key. Please check and try again.")
elif not sm.get_api_key("openai"):  # If no key is stored in sm and input is also empty
    st.warning("⚠️ Please enter your OpenAI API key to continue.")

app_is_configured = is_app_configured(sm)

st.subheader("Configuration Management")
# Import Configuration
uploaded_config_file = st.file_uploader(
    "Import Configuration File (JSON)",
    type=["json"],
    key="import_config_uploader",  # This widget key is important
    disabled=not app_is_configured,
    accept_multiple_files=False,  # Ensure only one config file
)

if uploaded_config_file is not None:
    try:
        loaded_config_data = json.load(uploaded_config_file)
        st.success(
            f"Configuration from '{uploaded_config_file.name}' ready to be applied."
        )

    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid configuration JSON.")
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        st.error(traceback.format_exc())

# Button to apply the loaded settings
if st.button(
    "📥 Apply Loaded Configuration",
    use_container_width=True,
    disabled=not app_is_configured or uploaded_config_file is None,
    key="apply_loaded_config_button",
):
    sm.load_app_configuration(
        loaded_config_data
    )  # Loads settings, updates relevant st.session_state widget keys

# Export Configuration
config_export_name = f"cleanrag_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

current_config_dict = (
    sm.get_app_configuration()
)  # API key is NOT included by default now
config_json = json.dumps(current_config_dict, indent=2)

st.download_button(
    label="💾 Export Current Configuration",
    data=config_json,
    file_name=config_export_name,
    use_container_width=True,
    disabled=not app_is_configured,
    mime="application/json",
    key="export_config_button",
)

# Model Settings Section
st.subheader("Model Settings")  # Changed to subheader

# Get current settings for selectbox indices
current_llm_model = sm.llm_model
current_embedding_model = sm.embedding_model


llm_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"]
# Get the current value from session state (which SM ensures is initialized or loaded)
current_llm_selection = st.session_state.llm_model # This key should be set by SM init or load

st.selectbox(
    "OpenAI Model",
    options=llm_options,
    # Calculate index based on the current session state value
    index=llm_options.index(current_llm_selection) if current_llm_selection in llm_options else 0,
    help="Select the OpenAI model to use.",
    key="llm_model", # This key is managed by SessionManager and this widget
    on_change=handle_llm_change,
    disabled=not app_is_configured,
)

embedding_options = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
current_embedding_selection = st.session_state.embedding_model

st.selectbox(
    "Embedding Model",
    options=embedding_options,
    index=embedding_options.index(current_embedding_selection) if current_embedding_selection in embedding_options else 0,
    help="Select the OpenAI embedding model to use.",
    key="embedding_model",
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
    disabled=not app_is_configured,
    key="file_uploader_widget",  # Use a distinct key for the widget
)
if uploaded_files_widget:
    sm.store_uploaded_files(uploaded_files_widget)  # Store them in session manager

# Processing options
st.subheader("Processing Options")

splitter_options = ["sentence", "semantic"]
current_splitter_selection = st.session_state.splitter_type

st.selectbox(
    "Splitter Type",
    options=splitter_options,
    index=splitter_options.index(current_splitter_selection) if current_splitter_selection in splitter_options else 0,
    format_func=lambda x: "Sentence-based" if x == "sentence" else "Semantic",
    help="Choose how to split documents.",
    key="splitter_type",
    on_change=handle_splitter_change,
    disabled=not app_is_configured,
)

active_splitter = st.session_state.splitter_type # Read from session state for conditional UI

if active_splitter == "sentence":
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Chunk Size", min_value=100, max_value=2000,
            value=st.session_state.chunk_size, # Directly use the session state value
            step=100,
            key="chunk_size",
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )
    with col2:
        st.slider(
            "Chunk Overlap", min_value=0, max_value=200,
            value=st.session_state.chunk_overlap, # Directly use the session state value
            step=10,
            key="chunk_overlap",
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )
else:  # semantic
    col1, col2 = st.columns(2)
    with col1:
        st.slider(
            "Buffer Size", min_value=1, max_value=5,
            value=st.session_state.buffer_size, # Directly use the session state value
            step=1,
            key="buffer_size",
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )
    with col2:
        st.slider(
            "Breakpoint Threshold", min_value=50, max_value=99,
            value=st.session_state.breakpoint_threshold, # Directly use the session state value
            step=1,
            key="breakpoint_threshold",
            on_change=handle_processing_options_change,
            disabled=not app_is_configured,
        )

vector_store_options = ["simple", "faiss"]
current_vector_store_selection = st.session_state.vector_store_type

st.selectbox(
    "Vector Store Type",
    options=vector_store_options,
    index=vector_store_options.index(current_vector_store_selection) if current_vector_store_selection in vector_store_options else 0,
    format_func=lambda x: "Simple" if x == "simple" else "FAISS",
    help="Choose the vector store type.",
    key="vector_store_type",
    on_change=handle_processing_options_change,
    disabled=not app_is_configured,
)

# Process button
stored_files_for_processing = sm.get_uploaded_files()  # Get latest from SM
if st.button(
    "🔍 Process Documents",
    type="primary",
    use_container_width=True,
    disabled=not app_is_configured or not stored_files_for_processing,
):
    sm.clear_processed_documents()  # Clears documents and index in SM
    if stored_files_for_processing:  # Check again due to button state
        # Ensure LlamaIndex settings are configured before processing documents
        configure_llama_index_settings(sm)
        progress_bar = st.progress(0, "Initializing document processing...")
        status_text = st.empty()
        total_files = len(stored_files_for_processing)

        for i, file_data in enumerate(stored_files_for_processing):
            actual_filename = (
                file_data.name if hasattr(file_data, "name") else f"File {i+1}"
            )
            status_text.text(f"Processing {actual_filename} ({i+1}/{total_files})...")
            try:
                text = extract_text_from_file(file_data)
                doc = Document(text=text, metadata={"filename": actual_filename})

                current_splitter_type = sm.splitter_type
                if current_splitter_type == "sentence":
                    kwargs = {
                        "chunk_size": sm.chunk_size,
                        "chunk_overlap": sm.chunk_overlap,
                    }
                else:  # semantic
                    kwargs = {
                        "buffer_size": sm.buffer_size,
                        "breakpoint_percentile_threshold": sm.breakpoint_threshold,
                    }

                chunks = process_document(
                    doc, splitter_type=current_splitter_type, **kwargs
                )  # Pass LlamaIndex Document to process_document

                stats = get_chunk_statistics(
                    chunks
                )  # Assumes chunks are LlamaIndex NodeWithScore or similar
                chunk_df = create_chunk_dataframe(chunks)

                document_data = {
                    "name": actual_filename,
                    "text": text,
                    "chunks": chunks,
                    "stats": stats,
                    "chunk_df": chunk_df,
                    "document": doc,
                }
                sm.store_processed_document(document_data)
                progress_bar.progress(
                    (i + 1) / total_files, f"Processed {actual_filename}"
                )

            except Exception as e:
                st.error(f"Error processing {actual_filename}: {str(e)}")

        status_text.text("Creating index from processed documents...")
        try:
            all_docs_for_index = [
                data["document"] for data in sm.get_all_processed_documents()
            ]
            if all_docs_for_index:
                if (
                    not Settings.embed_model
                ):  # Check if embed_model was successfully configured
                    st.error(
                        "Embedding model not configured. Cannot create index. Please check API key and model settings."
                    )
                else:
                    index = create_index_from_documents(
                        all_docs_for_index,
                        vector_store_type=sm.vector_store_type,
                        # embed_model will be taken from Settings.embed_model
                    )
                    sm.index = index  # Store the index in SessionManager
                    status_text.text("")  # Clear status text
                    progress_bar.empty()  # Clear progress bar
                    st.success("Documents processed and indexed successfully!")
                    st.rerun()  # Rerun to update UI, especially "Index Ready" status
            else:
                st.warning(
                    "No documents were successfully processed to create an index."
                )

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
            st.markdown("##### Document Statistics")  # Smaller heading
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", doc_data_item["stats"]["total_chunks"])
            with col2:
                st.metric(
                    "Avg Chunk Size",
                    f"{doc_data_item['stats']['avg_chunk_size_chars']:.1f} characters",
                )  # Assuming stats are char based
            with col3:
                st.metric(
                    "Total Words", doc_data_item["stats"]["total_words"]
                )  # Or characters if preferred

            st.markdown("##### Chunk Details")
            st.dataframe(
                doc_data_item["chunk_df"], use_container_width=True, height=200
            )

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
        if (
            COL_QUESTION not in new_qa_data.columns
            or COL_ANSWER_GROUND_TRUTH not in new_qa_data.columns
        ):
            st.error(
                f"QA file must contain '{COL_QUESTION}' and '{COL_ANSWER_GROUND_TRUTH}' columns."
            )
            # Optionally clear previously loaded data if new upload is invalid
            # sm.store_qa_data(None)
        else:
            prev_qa_data = sm.get_qa_data()
            if prev_qa_data is None or not new_qa_data.equals(prev_qa_data):
                sm.store_qa_data(new_qa_data)
                st.success(f"QA test set '{qa_file_widget.name}' loaded successfully.")
                if (
                    COL_CONTEXTS_GROUND_TRUTH in new_qa_data.columns
                    and new_qa_data[COL_CONTEXTS_GROUND_TRUTH].notna().any()
                ):
                    st.info(
                        f"Optional '{COL_CONTEXTS_GROUND_TRUTH}' column detected and will be used for evaluation."
                    )
                st.rerun()
    except Exception as e:
        st.error(f"Error reading or processing QA file: {e}")
        sm.store_qa_data(None)  # Clear QA data in session manager on error

# Display QA data if available
current_qa_data_df = sm.get_qa_data()
if current_qa_data_df is not None:
    st.markdown("##### Loaded QA Test Set (Preview)")
    st.dataframe(current_qa_data_df.head(), use_container_width=True)


# --- Chat Section ---
st.header("Chat 💬", anchor="chat")

if not sm.index:
    st.warning("⚠️ Chat functionality requires an index to work. Please ensure an index is available.")


with st.container(border=True):
    chat_history_display = sm.get_session_chat_history()
    for i, message_content in enumerate(chat_history_display):
        role = "user" if i % 2 == 0 else "assistant"
        with st.chat_message(role):
            st.write(message_content)

    user_query = st.chat_input(
        placeholder="Ask a question about your documents...",
        disabled=not app_is_configured
        or not sm.index,  # Disable if no API key or no index
    )

    if user_query:
        # Add user message to SM history *before* calling generate.
        # `generate_response` will then use this full history. (Adjust if model_utils expects otherwise)
        # sm.update_session_chat_history(user_query) # This makes generate_response's history param complex
        # Better: pass query, let generate_response use sm's history *before* this query

        # Ensure LlamaIndex settings are configured before chat
        try:
            configure_llama_index_settings(sm)
            if not Settings.llm:  # Check if LLM was successfully configured
                st.error(
                    "LLM not configured. Cannot chat. Please check API key and model settings."
                )
            else:
                assistant_response = generate_response(user_query, sm=sm)
                sm.update_session_chat_history(user_query)  # Store user query
                sm.update_session_chat_history(
                    assistant_response
                )  # Store assistant response
                st.rerun()
        except Exception as e:
            st.error(f"Error during chat: {e}")


if st.button(
    "Clear Chat History", use_container_width=True, disabled=not chat_history_display
):
    sm.clear_session_chat_history()
    st.rerun()

# --- Evaluate Section ---
st.header("Evaluate RAG Pipeline 📊", anchor="evaluate")

current_qa_data_df = sm.get_qa_data()
app_is_configured = is_app_configured(sm)

if current_qa_data_df is None or current_qa_data_df.empty or sm.index is None:
    st.warning("⚠️ Evaluation requires a QA file and an index to work. Please ensure both are available.")

st.subheader("Run New Evaluation Experiment")

# Input for experiment name
exp_name_default = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
experiment_name = st.text_input("Experiment Name", value=exp_name_default, disabled=not app_is_configured, key="experiment_name_input")

if st.button(
    "🧪 Run Evaluation Experiment",
    type="primary",
    use_container_width=True,
    disabled=current_qa_data_df is None
    or current_qa_data_df.empty
    or sm.index is None
    or not app_is_configured,
):
    if not experiment_name.strip():
        st.warning("Please provide a name for this evaluation experiment.")
    else:
        with st.spinner(
            f"Running Ragas evaluation for experiment: '{experiment_name}'... This may take a while."
        ):
            try:
                configure_llama_index_settings(sm) # Ensure LlamaIndex settings are current
                if not Settings.llm or not Settings.embed_model:
                    st.error("LlamaIndex LLM or Embedding Model not configured. Cannot run evaluation.")
                else:
                    openai_api_key = sm.get_api_key("openai")
                    if openai_api_key and not os.getenv("OPENAI_API_KEY"):
                        os.environ["OPENAI_API_KEY"] = openai_api_key
                    elif not openai_api_key and not os.getenv("OPENAI_API_KEY"):
                        st.warning("OpenAI API key not found. Ragas evaluation might fail.")

                    # Get current config for logging with results
                    config_summary = sm.get_app_configuration() 
                    # Remove sensitive data or simplify if needed for summary
                    # e.g., config_summary.pop("api_keys", None) is already handled by get_app_configuration

                    results_df = prepare_and_run_ragas_evaluation(sm)

                    if results_df is not None and not results_df.empty:
                        if 'ragas_evaluation_error' in results_df.columns:
                            st.error(f"Ragas evaluation for '{experiment_name}' encountered an error.")
                            st.dataframe(results_df) # Show data passed and error
                        else:
                            # Calculate overall ragas_score if individual scores are present
                            numeric_metric_cols = [m for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"] if m in results_df.columns]
                            overall_score = None
                            if numeric_metric_cols:
                                try:
                                    overall_score = results_df[numeric_metric_cols].mean(axis=0).mean() # Mean of mean of metrics
                                except pd.errors. soziale: # Handle non-numeric issues
                                    pass
                                
                            run_data = {
                                "name": experiment_name,
                                "timestamp": datetime.now().isoformat(),
                                "config_summary": config_summary,
                                "results_df_json": results_df.to_json(orient="records", indent=2), # Store DF as JSON string
                                "ragas_overall_score": overall_score if overall_score is not None else float('nan')
                            }
                            sm.add_evaluation_run(run_data)
                            st.success(f"Evaluation experiment '{experiment_name}' completed and results saved.")
                            st.rerun() # Rerun to update display of saved runs
                    else:
                        st.warning(f"Evaluation for '{experiment_name}' did not produce results.")
            except Exception as e:
                st.error(f"An error occurred during evaluation for '{experiment_name}': {e}")
                st.error(traceback.format_exc())

st.divider()
st.subheader("📜 Stored Evaluation Runs")
all_runs = sm.get_all_evaluation_runs()


run_names = [run["name"] for run in all_runs]

# --- Display and Compare Runs ---
selected_run_names = st.multiselect(
    "Select evaluation runs to display/compare:",
    options=run_names,
    default=run_names[-1:] if run_names else [], # Default to last run
    key="selected_eval_runs_multiselect"
)

if selected_run_names:
    for run_name in selected_run_names:
        run_data = sm.get_evaluation_run_by_name(run_name)
        if run_data:
            with st.expander(f"Results for Experiment: {run_data['name']} (Timestamp: {run_data['timestamp']})", expanded=len(selected_run_names) == 1):
                st.markdown("##### Configuration Used:")
                st.json(run_data['config_summary'], expanded=False)
                
                st.markdown("##### Ragas Scores:")
                
                if 'results_df_json' in run_data and run_data['results_df_json']:
                    try:
                        json_string_data = run_data['results_df_json']
                        results_df_from_json = pd.read_json(io.StringIO(json_string_data), orient="records")
                    except Exception as e_json_read:
                        st.error(f"Error parsing stored evaluation results for '{run_name}': {e_json_read}")
                        results_df_from_json = pd.DataFrame() # Empty DF on error
                else:
                    st.warning(f"No detailed Ragas scores found for run '{run_name}'.")
                    results_df_from_json = pd.DataFrame()
                                
                if 'ragas_overall_score' in run_data and not pd.isna(run_data['ragas_overall_score']):
                        st.metric("Mean Ragas Score", f"{run_data['ragas_overall_score']:.3f}")

                # Display individual metrics (similar to above)
                possible_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
                metrics_to_show = [m for m in possible_metrics if m in results_df_from_json.columns]
                
                if metrics_to_show:
                    metric_cols_display = st.columns(len(metrics_to_show))
                    for idx, metric_name in enumerate(metrics_to_show):
                        metric_series = pd.to_numeric(results_df_from_json[metric_name], errors='coerce')
                        if not metric_series.empty and metric_series.notna().any():
                            mean_score = metric_series.mean()
                            metric_cols_display[idx].metric(f"{metric_name.replace('_', ' ').title()}", f"{mean_score:.3f}")
                        else:
                            metric_cols_display[idx].metric(f"{metric_name.replace('_', ' ').title()}", "N/A")
                
                st.dataframe(results_df_from_json, use_container_width=True)

# --- Exporting ---
st.markdown("---")
st.markdown("#### Export Evaluation Runs")

export_filename_json = f"cleanrag_all_eval_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Prepare data for JSON export (convert DataFrames within each run to JSON strings if not already)
runs_for_export = []
for run_data_orig in all_runs:
    run_copy = run_data_orig.copy()
    if isinstance(run_copy.get('results_df'), pd.DataFrame): # Should not happen if stored as JSON string
        run_copy['results_df_json'] = run_copy.pop('results_df').to_json(orient="records", indent=2)
    runs_for_export.append(run_copy)
    
all_runs_json = json.dumps(runs_for_export, indent=2)

st.download_button(
    label="📦 Export All Evaluations",
    data=all_runs_json,
    file_name=export_filename_json,
    use_container_width=True,
    disabled=not all_runs,
    mime="application/json",
    key="export_all_eval_runs_button"
)

if st.button("🗑️ Clear All Evaluations", use_container_width=True, disabled=not all_runs, key="clear_all_eval_runs_button"):
    sm.clear_all_evaluation_runs()
    st.rerun()