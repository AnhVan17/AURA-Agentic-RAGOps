# Academic Chatbot 🎓

An advanced RAG (Retrieval-Augmented Generation) system for querying and summarizing academic papers using Google's Gemini models and Qdrant vector database.

## 🚀 Features

- **Document Processing**: Support for PDF, DOCX, and images using OCR (RapidOCR).
- **Advanced Retrieval**: Hybrid search combining Qdrant vector search and BM25 ranking.
- **Intelligent QA**: Powered by LangChain and Google Gemini (1.5 Flash/Pro).
- **Summarization**: Automated summarization of long academic documents.
- **Bilingual Support**: Optimized for Vietnamese and English.
- **Dual Interface**:
  - **REST API**: Built with FastAPI for integration.
  - **Interactive UI**: User-friendly Gradio interface.

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **LLM Framework**: LangChain, Google Generative AI
- **Vector Database**: Qdrant
- **OCR/Parsing**: RapidOCR, pdf2image, pdfminer.six
- **UI**: Gradio
- **Environment**: Python 3.10+

## 📋 Prerequisites

- Python 3.10 or higher
- Google AI Studio API Key (for Gemini)
- Poppler (for `pdf2image` functionality)

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd "academic chatbot"
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

## 🏃 Running the Application

### Start the API Backend
```bash
uvicorn app.main:app --reload
```
The API will be available at `http://localhost:8000`. Access the docs at `/docs`.

### Start the Gradio UI
```bash
python ui/gradio_app.py
```
The UI will be available at `http://localhost:7860`.

## 📂 Project Structure

- `app/`: FastAPI application (routes, schemas, middleware).
- `core/`: Core logic for embeddings, retrieval, and LLM orchestration.
- `ui/`: Gradio interface implementation.
- `lc/`: LangChain specific components and prompts.
- `data/`: Directory for source documents (excluded from Git).
- `qdrant_local_db/`: Persistent storage for the vector database.

## 📄 License
This project is for academic and research purposes.
