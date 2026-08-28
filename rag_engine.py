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

    def query(self, question: str, top_k: int = 8) -> List[Dict]:
        """
        Hybrid retrieval: combines FAISS semantic search, source filename matching,
        and automatic header chunk (Chunk 0) inclusion.

        Args:
            question: The user's question.
            top_k: Number of unique results to return.

        Returns:
            List of dicts with keys: text, source, chunk_id, score, images.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        # --- 1. Semantic search (FAISS) ---
        q_emb = self.embedding_model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")

        fetch_k = min(top_k * 4, self.index.ntotal)
        distances, indices = self.index.search(q_emb, fetch_k)

        semantic_results = {}
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata):
                semantic_results[idx] = float(dist)

        # --- 2. Source-Aware Keyword search (scan text + source filename) ---
        keyword_hits = self._keyword_search(question, top_k=top_k * 3)

        # --- 3. Combined score formula ---
        scored = {}
        all_indices = set(semantic_results.keys()).union(set(keyword_hits.keys()))

        for idx in all_indices:
            sem_dist = semantic_results.get(idx, 1.0)
            kw_rel = keyword_hits.get(idx, 0.0)

            # Boost formula: higher keyword relevance strongly reduces distance
            combined_score = sem_dist / (1.0 + 2.5 * kw_rel)
            scored[idx] = combined_score

        # --- 4. Sort and apply Header (Chunk 0) Auto-Inclusion ---
        sorted_indices = sorted(scored.keys(), key=lambda i: scored[i])

        # Identify top matched sources
        matched_sources = set()
        for idx in sorted_indices[:top_k]:
            if 0 <= idx < len(self.metadata):
                matched_sources.add(self.metadata[idx].get("source", ""))

        # Ensure Chunk 0 (header/title/author chunk) of every matched document is included
        header_indices = set()
        for i, meta in enumerate(self.metadata):
            src = meta.get("source", "")
            chunk_id = str(meta.get("chunk_id", ""))
            if src in matched_sources and (chunk_id.endswith("_0") or chunk_id == "0"):
                header_indices.add(i)

        # Combine header indices first, followed by score-ranked candidates
        final_candidates = list(header_indices) + [i for i in sorted_indices if i not in header_indices]

        results = []
        seen_texts = set()
        source_counts = {}
        max_per_source = max(3, top_k // 2)

        for idx in final_candidates:
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx].copy()
                src = meta.get("source", "unknown")
                text_fingerprint = meta.get("text", "")[:150].strip()

                if text_fingerprint in seen_texts:
                    continue
                seen_texts.add(text_fingerprint)
                meta["score"] = scored.get(idx, 0.5)

                curr_count = source_counts.get(src, 0)
                if curr_count < max_per_source or len(results) < top_k:
                    source_counts[src] = curr_count + 1
                    results.append(meta)
                    if len(results) >= top_k:
                        break

        return results

    def _keyword_search(self, question: str, top_k: int = 14) -> Dict[int, float]:
        """
        Keyword matching against chunk text and source filename.
        Returns dict of {chunk_index: relevance_score (0-1)}.
        """
        import re as _re

        # Extract meaningful keywords from the question (skip stopwords)
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
            "up", "get", "tell", "give", "also", "very", "just", "guy", "person",
        }
        words = _re.findall(r'\b[a-zA-Z0-9]+\b', question.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 1]

        if not keywords:
            return {}

        # Also extract multi-word phrases for better matching
        q_lower = question.lower()
        phrases = []
        phrase_patterns = [
            "program elective", "programme elective", "open elective",
            "5th sem", "5th semester", "semester v", "semester 5",
            "6th sem", "7th sem", "8th sem", "3rd sem", "4th sem",
            "semester vi", "semester vii", "semester viii",
            "semester iii", "semester iv", "semester i", "semester ii",
            "data science", "machine learning", "artificial intelligence",
            "web development", "full stack", "cloud computing",
            "full name", "projects listed", "projects",
        ]
        for phrase in phrase_patterns:
            if phrase in q_lower:
                phrases.append(phrase)

        # Score each chunk
        hits = {}
        for i, meta in enumerate(self.metadata):
            text_lower = meta.get("text", "").lower()
            source_lower = meta.get("source", "").lower()
            if not text_lower:
                continue

            # Combine source filename and text for keyword matching
            combined_target = f"{source_lower} {source_lower} {text_lower}"

            # Keyword hit ratio
            kw_hits = sum(1 for kw in keywords if kw in combined_target)
            kw_score = kw_hits / len(keywords) if keywords else 0

            # Match bonus if query keywords match words in document filename (e.g. "resume", "accenture")
            source_match_bonus = sum(0.4 for kw in keywords if kw in source_lower)

            # Phrase bonus (strong signal)
            phrase_bonus = sum(0.3 for p in phrases if p in combined_target)

            total_score = min(kw_score + source_match_bonus + phrase_bonus, 1.0)
            if total_score > 0.15:  # Minimum threshold
                hits[i] = total_score

        # Return top matches
        sorted_hits = sorted(hits.items(), key=lambda x: -x[1])[:top_k]
        return dict(sorted_hits)


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

        # System prompt — comprehensive, helpful, and smart
        system_prompt = (
            "You are an expert NMIT College AI Assistant with deep knowledge extraction skills.\n\n"
            "INSTRUCTIONS:\n"
            "1. Carefully read ALL the provided context chunks. The answer is very likely "
            "somewhere in the text — look for tables, lists, course codes, subject names, "
            "semester numbers, and any structured data.\n"
            "2. College documents often contain tables extracted as plain text. Rows may be "
            "split across lines. Course codes look like '22ISE561'. Pay attention to these patterns.\n"
            "3. Answer the question as COMPLETELY and HELPFULLY as possible. Use bullet points, "
            "tables, and clear formatting.\n"
            "4. If you find partial information, share what you found and mention what's missing.\n"
            "5. ONLY say you don't have information if you genuinely cannot find ANYTHING relevant "
            "in any of the context chunks.\n"
            "6. Reference which source numbers you used."
        )

        user_prompt = (
            f"CONTEXT (from ingested documents — read carefully, information may be in tables or lists):\n\n"
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