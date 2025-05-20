import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="CleanRAG - Diagnostics",
    page_icon="🔍",
    layout="wide"
)

st.title("Pipeline Diagnostics 🔍")

# Overview metrics
st.header("Pipeline Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Documents", "150")
with col2:
    st.metric("Total Chunks", "1,250")
with col3:
    st.metric("Avg. Retrieval Time", "0.23s")
with col4:
    st.metric("Success Rate", "92%")

# Retrieval Analysis
st.header("Retrieval Analysis")

# Sample data for visualization
retrieval_data = pd.DataFrame({
    'Query': [f'Query {i}' for i in range(10)],
    'Relevance Score': np.random.uniform(0.5, 1.0, 10),
    'Response Time': np.random.uniform(0.1, 0.5, 10),
    'Chunks Retrieved': np.random.randint(3, 8, 10)
})

# Relevance Score Distribution
fig1 = px.histogram(
    retrieval_data,
    x='Relevance Score',
    title='Relevance Score Distribution',
    nbins=20
)
st.plotly_chart(fig1, use_container_width=True)

# Response Time vs Relevance
fig2 = px.scatter(
    retrieval_data,
    x='Response Time',
    y='Relevance Score',
    title='Response Time vs Relevance Score',
    size='Chunks Retrieved'
)
st.plotly_chart(fig2, use_container_width=True)

# Mismatch Analysis
st.header("Mismatch Analysis")
st.markdown("""
### Common Issues Detected:
- **Irrelevant Retrievals**: 12 instances
- **Missing Context**: 5 instances
- **Duplicate Chunks**: 3 instances
- **Out-of-Context Answers**: 8 instances
""")

# Detailed Analysis
st.header("Detailed Analysis")
analysis_tab1, analysis_tab2 = st.tabs(["Query Analysis", "Chunk Analysis"])

with analysis_tab1:
    st.dataframe(retrieval_data)
    
with analysis_tab2:
    chunk_data = pd.DataFrame({
        'Chunk ID': [f'C{i}' for i in range(10)],
        'Usage Count': np.random.randint(1, 20, 10),
        'Avg. Relevance': np.random.uniform(0.5, 1.0, 10),
        'Source Document': [f'Doc {i%3 + 1}' for i in range(10)]
    })
    st.dataframe(chunk_data)

# Recommendations
st.header("Recommendations")
st.markdown("""
### Suggested Improvements:
1. **Chunk Size Optimization**
   - Current chunk size may be too large
   - Consider reducing to improve precision

2. **Embedding Model**
   - Current model shows good performance
   - No immediate changes needed

3. **Retrieval Strategy**
   - Consider implementing hybrid search
   - Add semantic similarity threshold

4. **Context Window**
   - Current window size is optimal
   - No changes recommended
""") 