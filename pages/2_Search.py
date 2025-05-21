import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from utils.data_utils import initialize_session_state

initialize_session_state()


st.set_page_config(
    page_title="CleanRAG - Search & Exploration", page_icon="🔎", layout="wide"
)

st.title("Search & Exploration 🔎")

# Search interface
st.header("Query Interface")
query = st.text_input("Enter your query", placeholder="Type your question here...")

# Search options
col1, col2 = st.columns(2)
with col1:
    num_results = st.slider("Number of results", 1, 10, 5)
    similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)

with col2:
    search_mode = st.radio(
        "Search Mode", ["Semantic", "Hybrid", "Keyword"], horizontal=True
    )
    rerank = st.checkbox("Enable reranking", value=True)

if st.button("Search", type="primary") and query:
    # Simulate search results
    results = pd.DataFrame(
        {
            "Chunk": [f"Chunk {i}" for i in range(num_results)],
            "Content": [
                f"This is the content of chunk {i} that matches your query."
                for i in range(num_results)
            ],
            "Similarity": np.random.uniform(similarity_threshold, 1.0, num_results),
            "Source": [f"Document {i%3 + 1}" for i in range(num_results)],
        }
    )

    # Display results
    st.header("Search Results")
    for idx, row in results.iterrows():
        with st.expander(f"Result {idx + 1} (Score: {row['Similarity']:.2f})"):
            st.markdown(f"**Source:** {row['Source']}")
            st.markdown(row["Content"])

    # Visualization
    st.header("Retrieval Analysis")

    # Similarity distribution
    fig1 = px.bar(
        results,
        x="Chunk",
        y="Similarity",
        title="Similarity Scores by Chunk",
        color="Similarity",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Source distribution
    fig2 = px.pie(results, names="Source", title="Results by Source Document")
    st.plotly_chart(fig2, use_container_width=True)

# Debugging Panel
st.header("Debugging Panel")
debug_tab1, debug_tab2 = st.tabs(["Query Analysis", "Retrieval Details"])

with debug_tab1:
    if query:
        st.markdown(
            """
        ### Query Analysis
        - **Query Type**: Question
        - **Key Terms**: term1, term2, term3
        - **Query Intent**: Information retrieval
        - **Expected Context**: Technical documentation
        """
        )

with debug_tab2:
    if query:
        st.markdown(
            """
        ### Retrieval Details
        - **Total Chunks Searched**: 1,250
        - **Search Time**: 0.23s
        - **Reranking Applied**: Yes
        - **Context Window**: 5 chunks
        """
        )
