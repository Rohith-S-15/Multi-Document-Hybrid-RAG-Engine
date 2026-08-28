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
    #  QUERY / RETRIEVAL (Hybrid: Semantic + Keyword)
    # ═══════════════════════════════════════════════════

    def query(self, question: str, top_k: int = 8, k_rrf: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF) Hybrid Retrieval.

        Combines vector semantic search ranks and clean keyword search ranks
        using the standard RRF formula: Score = 1/(60 + rank_sem) + 1/(60 + rank_kw).
        Completely parameter-free, robust across all document types (PDFs, resumes, syllabi).

        Args:
            question: The user's question.
            top_k: Number of unique results to return.
            k_rrf: RRF constant parameter (default 60).

        Returns:
            List of dicts with keys: text, source, chunk_id, page, score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # --- 1. Semantic search (FAISS) ranks ---
        q_emb = self.embedding_model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")

        fetch_k = min(max(top_k * 10, 50), self.index.ntotal)
        distances, indices = self.index.search(q_emb, fetch_k)

        sem_ranks = {}
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if 0 <= idx < len(self.metadata):
                sem_ranks[idx] = rank + 1

        # --- 2. Keyword search ranks ---
        kw_hits = self._keyword_search(question)
        kw_sorted = sorted(kw_hits.keys(), key=lambda i: -kw_hits[i])
        
        kw_ranks = {}
        for rank, idx in enumerate(kw_sorted):
            kw_ranks[idx] = rank + 1

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        candidate_indices = set(sem_ranks.keys()).union(set(kw_ranks.keys()))

        for idx in candidate_indices:
            r_sem = sem_ranks.get(idx, 10000)
            r_kw = kw_ranks.get(idx, 10000)
            rrf_score = (1.0 / (k_rrf + r_sem)) + (1.0 / (k_rrf + r_kw))
            rrf_scores[idx] = rrf_score

        # --- 4. Sort candidates by RRF Score ---
        sorted_by_rrf = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])

        results = []
        seen_fingerprints = set()
        source_counts = {}
        max_per_source = max(4, top_k // 2)

        for idx in sorted_by_rrf:
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                src = meta.get("source", "unknown")
                fp = meta.get("text", "")[:120].strip()

                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
                meta["score"] = rrf_scores[idx]

                curr_c = source_counts.get(src, 0)
                if curr_c < max_per_source or len(results) < top_k:
                    source_counts[src] = curr_c + 1
                    results.append(meta)
                    if len(results) >= top_k:
                        break

        return results

    def _keyword_search(self, question: str) -> Dict[int, float]:
        """
        Clean keyword matching against chunk text.
        Returns dict of {chunk_index: keyword_match_count}.
        """
        import re as _re

        stopwords = {
            "i", "me", "my", "we", "our", "you", "your", "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "shall", "can", "may", "might", "am", "it", "its", "this", "that",
            "these", "those", "in", "on", "at", "to", "for", "of", "with",
            "from", "by", "as", "into", "about", "between", "through",
            "and", "or", "but", "not", "no", "if", "then", "so", "than",
            "what", "which", "who", "whom", "how", "when", "where", "why",
            "all", "each", "every", "any", "some", "gonna", "going", "take",
            "up", "get", "tell", "give", "also", "very", "just",
        }
        words = _re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        if not keywords:
            return {}

        hits = {}
        for i, meta in enumerate(self.metadata):
            text_lower = meta.get("text", "").lower()
            if not text_lower:
                continue

            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                hits[i] = float(score)

        return hits


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
            page_info = f" (Page {chunk.get('page')})" if chunk.get("page") else ""
            text = chunk.get("text", "")
            context_parts.append(f"[Source {i + 1}: {source}{page_info}]\n{text}")
        context = "\n\n".join(context_parts)

        # System prompt — comprehensive, helpful, and smart
        system_prompt = (
            "You are an expert AI Research & Information Assistant with deep document comprehension skills.\n\n"
            "INSTRUCTIONS:\n"
            "1. Carefully read ALL the provided context chunks. Context chunks come from uploaded PDFs, resumes, research papers, and documents.\n"
            "2. Search for exact facts, definition tables, methodology steps, numbers, dataset details, course codes, and author details.\n"
            "3. Answer the question as COMPLETELY, ACCURATELY, and HELPFULLY as possible using bullet points, tables, and clear Markdown formatting.\n"
            "4. Clearly cite which Source numbers (and Page numbers if available) you retrieved each piece of information from.\n"
            "5. ONLY say you don't have information if NO provided context chunk contains relevant details."
        )

        user_prompt = (
            f"CONTEXT (from ingested documents):\n\n"
            f"{context}\n\n"
            f"QUESTION: {question}\n\n"
            f"Provide a thorough, well-formatted answer:"
        )

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
        from document_processor import load_document
        print(f"[RAG] Ingesting document: {filepath}")
        chunks = load_document(filepath)
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