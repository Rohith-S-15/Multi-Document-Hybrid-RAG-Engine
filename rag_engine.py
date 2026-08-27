"""
rag_engine.py — Core RAG engine for the College RAG Chatbot.

Handles: FAISS vector storage, sentence-transformer embeddings,
         Groq LLM integration, document/URL ingestion, and answer generation.

Architecture (inspired by Krishna Naik's RAG Tutorial):
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2
  - Vector Store: FAISS (Facebook AI Similarity Search)
  - LLM: Groq API (gemma2-9b-it) via LangChain
  - Chunking: LangChain RecursiveCharacterTextSplitter

Integration contract:
  - RagEngine.ingest_document(filepath)       → ingest a file
  - RagEngine.ingest_url(url)                 → scrape + ingest a URL
  - RagEngine.query(question, top_k)          → retrieve relevant chunks
  - RagEngine.generate_answer(question, chunks) → generate LLM answer
  - RagEngine.get_stats()                     → index statistics
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is required. Install with: pip install faiss-cpu"
    )


class RagEngine:
    """
    Production-ready RAG engine using FAISS + Groq + sentence-transformers.
    """

    def __init__(
        self,
        persist_dir: str = "./faiss_store",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        groq_api_key: str = None,
        llm_model: str = "qwen/qwen3.6-27b",
    ):
        """
        Initialize the RAG engine.

        Args:
            persist_dir: Directory to save/load the FAISS index.
            embedding_model_name: Sentence-transformers model name.
            groq_api_key: Groq API key (or set GROQ_API_KEY env var).
            llm_model: Groq model name for chat completion.
        """
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        # Initialize embedding model
        print(f"[RAG] Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_embedding_dimension()

        # Initialize FAISS index
        self.index = None
        self.metadata = []  # List of dicts: text, source, chunk_id, images
        self._load_or_create_index()

        # Initialize Groq LLM
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY must be set in environment or passed to RagEngine. "
                "Get a free key at https://console.groq.com/"
            )
        self.llm_model_name = llm_model
        self.llm = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=llm_model,
        )
        print(f"[RAG] Groq LLM initialized: {llm_model}")

    # ═══════════════════════════════════════════════════
    #  FAISS INDEX MANAGEMENT
    # ═══════════════════════════════════════════════════

    def _load_or_create_index(self) -> None:
        """Load existing FAISS index from disk, or create a new one."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(faiss_path)
                with open(meta_path, "rb") as f:
                    self.metadata = pickle.load(f)
                print(f"[RAG] Loaded existing index: {self.index.ntotal} vectors")
            except Exception as e:
                print(f"[RAG] Failed to load index, creating new: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a fresh FAISS index."""
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.metadata = []
        print("[RAG] Created new FAISS index")

    def _save_index(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[RAG] Saved index: {self.index.ntotal} vectors")

    # ═══════════════════════════════════════════════════
    #  EMBEDDING & INDEXING
    # ═══════════════════════════════════════════════════

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Embed and add text chunks to the FAISS index.

        Each chunk dict should have keys: 'text', 'source', 'chunk_id'.
        Optional key: 'images' (list of image URLs).
        """
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        print(f"[RAG] Embedding {len(texts)} chunks...")

        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts, convert_to_numpy=True, show_progress_bar=len(texts) > 10
        ).astype("float32")

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata
        self.metadata.extend(chunks)

        # Persist to disk
        self._save_index()
        print(f"[RAG] Added {len(chunks)} chunks. Total: {self.index.ntotal}")

    # ═══════════════════════════════════════════════════
    #  QUERY / RETRIEVAL
    # ═══════════════════════════════════════════════════

    def query(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve the most relevant chunks for a question.

        Args:
            question: The user's question.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, source, chunk_id, score, images.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # Embed the query
        q_emb = self.embedding_model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")

        # Search FAISS (returns distances and indices)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_emb, k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                meta["score"] = float(dist)
                results.append(meta)

        return results

    # ═══════════════════════════════════════════════════
    #  ANSWER GENERATION (Groq LLM)
    # ═══════════════════════════════════════════════════

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Generate an answer using Groq LLM with retrieved context.

        Args:
            question: The user's question.
            context_chunks: Retrieved chunks from the vector store.

        Returns:
            LLM-generated answer string.
        """
        if not context_chunks:
            return "I don't have enough information to answer that question."

        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Source {i + 1}: {source}]\n{text}")
        context = "\n\n".join(context_parts)

        # System prompt for grounded answers
        system_prompt = (
            "You are an official NMIT College AI Assistant. "
            "Answer the user's question accurately based ONLY on the provided context.\n"
            "If the answer cannot be determined from the context, state clearly: "
            "'I don't have this information in the ingested college documents.'\n"
            "Format your answer clearly with bullet points if helpful, "
            "and reference the source numbers used."
        )

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            answer = response.content

            # Strip Qwen3.6 <think>...</think> reasoning tags
            import re
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

            return answer
        except Exception as e:
            print(f"[RAG] Groq API error: {e}")
            return f"Error generating response: {e}"

    # ═══════════════════════════════════════════════════
    #  INGESTION METHODS
    # ═══════════════════════════════════════════════════

    def ingest_document(self, filepath: str) -> None:
        """Ingest a document file (PDF, DOCX, TXT, CSV) into the RAG index."""
        from document_processor import load_document, chunk_text
        print(f"[RAG] Ingesting document: {filepath}")
        text = load_document(filepath)
        source_name = os.path.basename(filepath)
        chunks = chunk_text(text, source_name=source_name)
        self.add_chunks(chunks)

    def ingest_url(self, url: str) -> None:
        """Scrape a URL and ingest the content into the RAG index."""
        from web_scraper import scrape_and_chunk
        print(f"[RAG] Ingesting URL: {url}")
        chunks = scrape_and_chunk(url)
        self.add_chunks(chunks)

    # ═══════════════════════════════════════════════════
    #  STATISTICS
    # ═══════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        """Get statistics about the current index."""
        sources = list(set(m.get("source", "") for m in self.metadata))
        return {
            "total_chunks": self.index.ntotal if self.index else 0,
            "unique_sources": len(sources),
            "sources": sources,
        }

    def clear_index(self) -> None:
        """Clear the entire index and start fresh."""
        self._create_new_index()
        self._save_index()
        print("[RAG] Index cleared")