# CleanRAG

**CleanRAG** is an interactive tool built with Streamlit, designed to help developers and researchers optimize their Retrieval-Augmented Generation (RAG) pipelines. The application provides a clean UI for uploading documents, configuring chunking and indexing strategies, and evaluating performance—all within a streamlined environment.

## 🚀 Key Features

- Intuitive interface for document upload and configuration  
- Support for multiple text splitting and indexing strategies  
- Integration with OpenAI language and embedding models  
- Evaluation of RAG performance using the RAGAS framework  
- Side-by-side configuration comparison for iterative optimization  

## 🧠 Technical Highlights

- Built with **Streamlit** for rapid interface development  
- Indexing powered by **LlamaIndex** for flexible, ETL-style text chunking and retrieval  
- Model interaction and embeddings via **OpenAI APIs**  
- Evaluation using **RAGAS metrics**, including:
  - Faithfulness  
  - Context utilization  
  - Answer relevance  

## Project Structure

- `app.py`: Main application entry point  
- `config.py`: Configuration logic for app settings  
- `utils/`: Helper functions for chunking, indexing, and evaluation  
- `requirements.txt`: Python package dependencies  
- `README.md`: Project documentation  

## Setup Instructions

1. Ensure you have Python installed  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser to `http://localhost:8501` to interact with the app

## Dependencies

Core libraries include:
- `streamlit`
- `llama-index`
- `openai`
- `ragas`
- `pandas`, `numpy`
