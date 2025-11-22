# Multimodal RAG System

A robust Retrieval-Augmented Generation (RAG) system capable of processing and querying financial documents containing both text and images. This system leverages advanced NLP techniques, including multimodal embeddings, cross-encoder re-ranking, and local LLM integration to provide accurate and context-aware responses.

## 🚀 Features

-   **Multimodal Retrieval**: Extracts and indexes both text and images from PDF documents.
-   **Hybrid Search**: Combines semantic search with keyword boosting for high-precision retrieval.
-   **Re-ranking**: Utilizes a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to filter and rank retrieved chunks, ensuring the LLM receives the most relevant context.
-   **Local LLM Support**: Integrated with `google/flan-t5-large` for privacy-preserving, offline inference (with OpenAI API fallback).
-   **Interactive UI**: A user-friendly Streamlit interface for chatting with your documents and visualizing results.
-   **Evaluation Tools**: Scripts to visualize the embedding space using t-SNE.

## 🛠️ Tech Stack

-   **Language**: Python 3.8+
-   **Interface**: Streamlit
-   **Embeddings**: Sentence-BERT (`all-MiniLM-L6-v2`), CLIP (`clip-ViT-B-32`)
-   **Vector DB**: FAISS
-   **LLM**: Google Flan-T5 (Local) / OpenAI GPT-3.5 (Optional)
-   **PDF Processing**: PyMuPDF (Fitz)

## 📂 Project Structure

```
.
├── app.py                 # Streamlit User Interface
├── rag_core.py            # Core RAG logic (Retrieval, Generation, Re-ranking)
├── evaluate.py            # Visualization and Evaluation scripts
├── diagnose_rag.py        # Diagnose RAG system
├── verify_data.py         # Verify data processing
├── requirements.txt       # Python dependencies
├── scripts/
│   └── text-extraction/   # Data processing scripts
│       ├── text_extraction.py
│       ├── Data/          # Source PDFs
│       └── output/        # Generated embeddings and index
└── README.md
```

## ⚡ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd genAI-A3
    ```

2.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install streamlit
    ```

## 🏃‍♂️ Usage

### 1. Data Preparation
If you have new PDFs, place them in `scripts/text-extraction/Data/` and run the extraction pipeline:
```bash
cd scripts/text-extraction
python text_extraction.py
cd ../..
```
*Note: This will generate the embeddings and FAISS index in `scripts/text-extraction/output/`.*

### 2. Running the Application
Start the Streamlit web interface:
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

### 3. Configuration
-   **OpenAI API**: You can enter your API key in the sidebar to use GPT-3.5.
-   **Local LLM**: If no key is provided, the system defaults to `flan-t5-large` (running locally on CPU).
-   **Top-K**: Adjust the number of retrieved chunks in the sidebar.

## 📊 Evaluation
To visualize the embedding space of your documents:
```bash
python evaluate.py
```
This generates `embedding_visualization.png` showing the clusters of text chunks.

## 🧠 How It Works

1.  **Ingestion**: PDFs are parsed; text is chunked, and images are extracted.
2.  **Embedding**: Text and images are converted into vector embeddings.
3.  **Retrieval**:
    -   **Stage 1**: FAISS retrieves the top-40 candidates based on semantic similarity and keyword boosting.
    -   **Stage 2**: A Cross-Encoder re-ranks these candidates to find the top-5 most relevant chunks.
4.  **Generation**: The selected chunks are passed as context to the LLM to generate the final answer.

