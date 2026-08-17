# College RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded college documents or scraped college websites, with source citations.

## Tech Stack

- **LLM**: Hugging Face Inference API (Mistral-7B-Instruct-v0.2)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB (local persistence)
- **UI**: Gradio
- **File Parsing**: PyPDF2, python-docx
- **Web Scraping**: BeautifulSoup4, requests

## Project Structure

- `app.py` – Gradio interface
- `rag_engine.py` – Core RAG logic (embedding, retrieval, LLM call)
- `document_processor.py` – Load and chunk PDF/DOCX/TXT files
- `web_scraper.py` – Scrape college websites and convert to text chunks
- `requirements.txt` – Python dependencies

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your Hugging Face API token as an environment variable:
   ```bash
   set HF_TOKEN=your_hf_token
   ```
   (Linux/macOS: `export HF_TOKEN=your_hf_token`)
4. Run the app: `python app.py`

## Division of Labor (Fable vs Opus)

- **Fable**: `rag_engine.py`, `app.py`
- **Opus**: `document_processor.py`, `web_scraper.py`

## License

MIT