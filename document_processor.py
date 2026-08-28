"""
document_processor.py — File parsing and text chunking for the College RAG Chatbot.

Owner: Opus (Antigravity)
Handles: PDF, DOCX, TXT, CSV extraction + intelligent text chunking with overlap.

Uses LangChain document loaders for robust file handling and
RecursiveCharacterTextSplitter for context-aware chunking.

Integration contract:
  - load_document(filepath) → str               (used by rag_engine.ingest_document)
  - chunk_text(text, source_name) → List[Dict]   (used by rag_engine + web_scraper)
"""

import os
import re
from pathlib import Path
from typing import List, Dict


# ═══════════════════════════════════════════════════════
#  DOCUMENT LOADING (LangChain-powered)
# ═══════════════════════════════════════════════════════

def load_document(filepath: str) -> List[Dict]:
    """
    Load text content from a file using LangChain document loaders.
    Returns a list of chunk dicts: [{'text': ..., 'source': ..., 'chunk_id': ..., 'page': ...}].
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = Path(filepath).suffix.lower()
    source_name = os.path.basename(filepath)

    if ext == ".pdf":
        return _load_pdf_page_aware(filepath, source_name=source_name)
    elif ext == ".docx":
        raw_text = _load_docx(filepath)
        return chunk_text(raw_text, source_name=source_name)
    elif ext == ".txt":
        raw_text = _load_txt(filepath)
        return chunk_text(raw_text, source_name=source_name)
    elif ext == ".csv":
        raw_text = _load_csv(filepath)
        return chunk_text(raw_text, source_name=source_name)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: .pdf, .docx, .txt, .csv"
        )


def _load_pdf_page_aware(filepath: str, source_name: str, chunk_size: int = 700, overlap: int = 150) -> List[Dict]:
    """Extract and split text page-by-page from a PDF to maintain page-aware granularity."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(str(filepath))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        page_num = doc.metadata.get("page", 0) + 1
        page_text = _clean_text(doc.page_content.strip())
        if not page_text:
            continue

        page_splits = splitter.split_text(page_text)
        for j, split_str in enumerate(page_splits):
            split_str = split_str.strip()
            if len(split_str) < 40:
                continue
            chunks.append({
                "text": f"[Page {page_num}]\n{split_str}",
                "source": source_name,
                "chunk_id": f"{source_name}_p{page_num}_{j}",
                "page": page_num,
            })

    return chunks


def _load_docx(filepath: str) -> str:
    """Extract text from a DOCX using LangChain's Docx2txtLoader."""
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(filepath))
    docs = loader.load()
    text = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())
    return _clean_text(text)


def _load_txt(filepath: str) -> str:
    """Load a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return _clean_text(f.read())


def _load_csv(filepath: str) -> str:
    """Extract text from a CSV using LangChain's CSVLoader."""
    from langchain_community.document_loaders import CSVLoader
    loader = CSVLoader(str(filepath))
    docs = loader.load()
    rows = [doc.page_content for doc in docs if doc.page_content.strip()]
    return _clean_text("\n\n".join(rows))


def _clean_text(text: str) -> str:
    """Clean up extracted text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════
#  TEXT CHUNKING
# ═══════════════════════════════════════════════════════

def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 700,
    overlap: int = 150,
    images: List[str] = None,
) -> List[Dict[str, str]]:
    """
    Split non-PDF text into overlapping chunks using RecursiveCharacterTextSplitter.
    """
    if not text or not text.strip():
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    split_texts = splitter.split_text(text)

    chunks = []
    for i, chunk_str in enumerate(split_texts):
        chunk_str = chunk_str.strip()
        if len(chunk_str) < 40:
            continue
        chunks.append({
            "text": chunk_str,
            "source": source_name,
            "chunk_id": f"{source_name}_{i}",
            "images": images or [],
        })

    return chunks


# ═══════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════

def get_supported_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return [".pdf", ".docx", ".txt", ".csv"]


def get_document_stats(filepath: str) -> Dict:
    """
    Get basic stats about a document without fully processing it.

    Returns:
        Dict with keys: filename, extension, size_bytes, size_human.
    """
    path = Path(filepath)
    size = path.stat().st_size

    # Human-readable size
    if size < 1024:
        size_human = f"{size} B"
    elif size < 1024 * 1024:
        size_human = f"{size / 1024:.1f} KB"
    else:
        size_human = f"{size / (1024 * 1024):.1f} MB"

    return {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": size,
        "size_human": size_human,
    }