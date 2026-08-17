"""
document_processor.py — File parsing and text chunking for the College RAG Chatbot.

Owner: Opus (Antigravity)
Handles: PDF, DOCX, TXT extraction + intelligent text chunking with overlap.

Integration contract:
  - load_document(filepath) → str          (used by rag_engine.ingest_document)
  - chunk_text(text, source_name) → List[Dict]  (used by rag_engine + web_scraper)
"""

import os
import re
from pathlib import Path
from typing import List, Dict

import PyPDF2
import docx


# ═══════════════════════════════════════════════════════
#  DOCUMENT LOADING
# ═══════════════════════════════════════════════════════

def load_document(filepath: str) -> str:
    """
    Load text content from a PDF, DOCX, or TXT file.

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

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_txt,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: {', '.join(loaders.keys())}"
        )

    raw_text = loader(filepath)
    return _clean_text(raw_text)


def _load_pdf(filepath: str) -> str:
    """Extract text from a PDF, preserving page markers."""
    pages = []
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[Page {i + 1}]\n{page_text}")
    return "\n\n".join(pages)


def _load_docx(filepath: str) -> str:
    """Extract text from a DOCX file."""
    doc = docx.Document(filepath)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n\n".join(paragraphs)


def _load_txt(filepath: str) -> str:
    """Load a plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


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
#  TEXT CHUNKING
# ═══════════════════════════════════════════════════════

def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks using word boundaries.

    This avoids cutting words in half (which char-based chunking does).
    Each chunk preserves context via overlap with the previous chunk.

    Args:
        text:        The full text to split.
        source_name: The source identifier (filename or URL).
        chunk_size:  Approximate number of words per chunk.
        overlap:     Number of words to overlap between consecutive chunks.

    Returns:
        List of dicts, each with keys: 'text', 'source', 'chunk_id'.

    Example:
        >>> chunks = chunk_text("Hello world ...", "notes.pdf", chunk_size=100)
        >>> chunks[0]
        {'text': 'Hello world ...', 'source': 'notes.pdf', 'chunk_id': 'notes.pdf_0'}
    """
    if not text or not text.strip():
        return []

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words).strip()

        # Only add non-trivial chunks (at least 20 words)
        if len(chunk_words) >= 20:
            chunks.append({
                "text": chunk_str,
                "source": source_name,
                "chunk_id": f"{source_name}_{chunk_id}",
            })
            chunk_id += 1

        # Move forward by (chunk_size - overlap) words
        step = max(chunk_size - overlap, 1)
        start += step

    return chunks


# ═══════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════

def get_supported_extensions() -> List[str]:
    """Return list of supported file extensions."""
    return [".pdf", ".docx", ".txt"]


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