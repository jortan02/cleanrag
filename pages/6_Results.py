import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(
    page_title="CleanRAG - Results & Visualization",
    page_icon="📊",
    layout="wide"
)

st.title("QA Test Results & Visualization 📊")

# QA Test Overview
st.header("QA Test Overview")

# Generate sample QA test data
qa_data = pd.DataFrame({
    'Question': [f'Q{i}' for i in range(20)],
    'Correct Answer': [f'Answer {i}' for i in range(20)],
    'Retrieved Answer': [f'Retrieved {i}' for i in range(20)],
    'Similarity Score': np.random.uniform(0.5, 1.0, 20),
    'Retrieval Time': np.random.uniform(0.1, 0.5, 20),
    'Chunks Used': np.random.randint(1, 5, 20),
    'Category': np.random.choice(['Technical', 'General', 'Specific', 'Complex'], 20)
})

# Performance Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Answer Accuracy",
        f"{np.mean(qa_data['Similarity Score']):.1%}",
        f"{np.std(qa_data['Similarity Score']):.1%}"
    )

with col2:
    st.metric(
        "Avg Response Time",
        f"{qa_data['Retrieval Time'].mean():.2f}s",
        f"{qa_data['Retrieval Time'].std():.2f}s"
    )

with col3:
    st.metric(
        "Avg Chunks Used",
        f"{qa_data['Chunks Used'].mean():.1f}",
        f"{qa_data['Chunks Used'].std():.1f}"
    )

with col4:
    correct_answers = np.sum(qa_data['Similarity Score'] > 0.8)
    st.metric(
        "High Confidence Answers",
        f"{correct_answers}",
        f"{correct_answers/len(qa_data):.1%}"
    )

# Answer Quality Analysis
st.header("Answer Quality Analysis")

# Create tabs for different analyses
analysis_tabs = st.tabs(["Answer Distribution", "Category Performance", "Detailed Results"])

with analysis_tabs[0]:
    # Similarity Score Distribution
    fig1 = px.histogram(
        qa_data,
        x='Similarity Score',
        title='Answer Similarity Distribution',
        nbins=20,
        color_discrete_sequence=['#1f77b4']
    )
    st.plotly_chart(fig1, use_container_width=True)

with analysis_tabs[1]:
    # Category Performance
    category_performance = qa_data.groupby('Category').agg({
        'Similarity Score': 'mean',
        'Retrieval Time': 'mean',
        'Chunks Used': 'mean'
    }).reset_index()
    
    fig2 = px.bar(
        category_performance,
        x='Category',
        y='Similarity Score',
        title='Performance by Question Category',
        color='Similarity Score',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig2, use_container_width=True)

with analysis_tabs[2]:
    # Detailed Results Table
    st.dataframe(
        qa_data[['Question', 'Correct Answer', 'Retrieved Answer', 'Similarity Score', 'Category']],
        use_container_width=True
    )

# Chunk Usage Analysis
st.header("Chunk Usage Analysis")

# Generate chunk usage data
chunk_usage = pd.DataFrame({
    'Chunk ID': [f'C{i}' for i in range(15)],
    'Usage Count': np.random.randint(1, 20, 15),
    'Avg. Similarity': np.random.uniform(0.5, 1.0, 15),
    'Category': np.random.choice(['Technical', 'General', 'Specific', 'Complex'], 15)
})

# Chunk Usage vs Similarity
fig3 = px.scatter(
    chunk_usage,
    x='Usage Count',
    y='Avg. Similarity',
    size='Usage Count',
    color='Category',
    title='Chunk Usage vs Answer Similarity',
    hover_data=['Chunk ID']
)
st.plotly_chart(fig3, use_container_width=True)

# Performance by Question Type
st.header("Performance by Question Type")

# Create radar chart for category performance
fig4 = go.Figure()

for category in qa_data['Category'].unique():
    category_data = qa_data[qa_data['Category'] == category]
    fig4.add_trace(go.Scatterpolar(
        r=[
            category_data['Similarity Score'].mean(),
            category_data['Retrieval Time'].mean(),
            category_data['Chunks Used'].mean()
        ],
        theta=['Similarity', 'Response Time', 'Chunks Used'],
        name=category
    ))

fig4.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    title='Performance Metrics by Question Category'
)

st.plotly_chart(fig4, use_container_width=True)

# Export Options
st.header("Export Options")
export_format = st.selectbox(
    "Select Export Format",
    ["CSV", "JSON", "PDF"]
)

if st.button("Export Results", type="primary"):
    st.success(f"QA test results exported successfully in {export_format} format!") 