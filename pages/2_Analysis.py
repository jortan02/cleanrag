import streamlit as st
import pandas as pd
import numpy as np
from config import APP_NAME, APP_ICON

st.set_page_config(
    page_title=f"{APP_NAME} - Analysis",
    page_icon=APP_ICON,
    layout="wide"
)

st.title("Data Analysis")
st.markdown("""
This page demonstrates various data analysis capabilities:
- Data upload and preview
- Basic statistics
- Interactive visualizations
- Data filtering and exploration
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the data
    df = pd.read_csv(uploaded_file)
    
    # Display basic information
    st.subheader("Data Preview")
    st.dataframe(df.head())
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())
    
    # Column selection for visualization
    st.subheader("Data Visualization")
    selected_column = st.selectbox("Select a column to visualize", df.columns)
    
    # Create visualization based on data type
    if pd.api.types.is_numeric_dtype(df[selected_column]):
        st.line_chart(df[selected_column])
    else:
        st.bar_chart(df[selected_column].value_counts())
else:
    st.info("Please upload a CSV file to begin analysis.") 