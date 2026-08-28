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

def load_document(filepath: str) -> str:
    """
    Load text content from a file using LangChain document loaders.

    Supported formats: PDF, DOCX, TXT, CSV.

    Args:
        filepath: Path to the document file.

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = Path(filepath).suffix.lower()

    try:
        if ext == ".pdf":
            return _load_pdf(filepath)
        elif ext == ".docx":
            return _load_docx(filepath)
        elif ext == ".txt":
            return _load_txt(filepath)
        elif ext == ".csv":
            return _load_csv(filepath)
        else:
            raise ValueError(
                f"Unsupported file type: '{ext}'. Supported: .pdf, .docx, .txt, .csv"
            )
    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        print(f"[document_processor] Error loading {filepath}: {e}")
        # Fallback: try reading as plain text
        try:
            return _load_txt(filepath)
        except Exception:
            raise ValueError(f"Failed to load {filepath}: {e}")


def _load_pdf(filepath: str) -> str:
    """Extract text from a PDF using LangChain's PyPDFLoader."""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(filepath))
    docs = loader.load()
    pages = []
    for doc in docs:
        text = doc.page_content.strip()
        if text:
            page_num = doc.metadata.get("page", "?")
            pages.append(f"[Page {int(page_num) + 1}]\n{text}")
    return _clean_text("\n\n".join(pages))


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
    """
    Clean up extracted text:
    - Collapse multiple blank lines → single blank line
    - Collapse multiple spaces → single space
    - Strip leading/trailing whitespace
    """
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces (but not newlines) into one
    text = re.sub(r"[^\S\n]+", " ", text)
    return text.strip()


# ═══════════════════════════════════════════════════════
#  TEXT CHUNKING (LangChain RecursiveCharacterTextSplitter)
# ═══════════════════════════════════════════════════════

def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 500,
    overlap: int = 100,
    images: List[str] = None,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

    Uses character-based splitting with intelligent separators that respect
    paragraph, sentence, and word boundaries.

    Args:
        text:        The full text to split.
        source_name: The source identifier (filename or URL).
        chunk_size:  Approximate number of characters per chunk.
        overlap:     Number of characters to overlap between consecutive chunks.
        images:      Optional list of image URLs associated with this source.

    Returns:
        List of dicts, each with keys: 'text', 'source', 'chunk_id', 'images'.
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
        if len(chunk_str) < 50:  # Skip tiny chunks
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