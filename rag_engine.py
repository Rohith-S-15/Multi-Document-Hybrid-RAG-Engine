import os
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import chromadb
import requests
from huggingface_hub import InferenceClient

class RagEngine:
    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "college_rag",
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hf_api_url: str = None,
                 hf_token: str = None):
        """
        Initialize the RAG engine.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model_name)
        # Initialize Chroma persistent client
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(name=collection_name)
        # Hugging Face Inference API settings
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN must be set in environment or passed to RagEngine")
        # Default to Qwen/Qwen2.5-7B-Instruct if not provided
        self.model_name = hf_api_url or os.getenv("HF_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
        self.inference_client = InferenceClient(token=self.hf_token)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return list of embeddings."""
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: List[Dict[str, str]]) -> None:
        """
        Add chunks to the vector store.
        Each chunk dict must have keys: 'text', 'source', 'chunk_id'.
        """
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self._embed_texts(texts)
        # Prepare data for Chroma
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"source": c["source"], "text": c["text"]} for c in chunks]
        self.collection.add(
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def query(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Query the vector store for similar chunks.
        Returns list of dicts with keys: text, source, chunk_id, score.
        """
        q_emb = self._embed_texts([question])[0]
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        # Chroma returns distances; we can convert to similarity (1 - distance) if using cosine?
        # For simplicity, we'll return distance as score.
        outputs = []
        if results['ids'] and results['ids'][0]:
            for idx, chunk_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][idx]
                distance = results['distances'][0][idx] if 'distances' in results else None
                outputs.append({
                    "chunk_id": chunk_id,
                    "text": metadata.get("text", ""),
                    "source": metadata.get("source", ""),
                    "score": distance  # lower is better
                })
        return outputs

    def _call_hf_api(self, prompt: str, system_prompt: str = None, max_new_tokens: int = 500, temperature: float = 0.7) -> str:
        """Call Hugging Face Inference API using chat_completion."""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.inference_client.chat_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"HF API error: {e}")
            return f"Error generating response: {e}"

    def generate_answer(self, question: str, context_chunks: List[Dict]) -> str:
        """
        Build prompt with context and call LLM.
        """
        if not context_chunks:
            return "I don't have enough information to answer that question."
        # Build context string
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk.get('source', 'unknown')
            text = chunk.get('text', '')
            context_parts.append(f"[Source {i+1}: {source}]\n{text}")
        context = "\n\n".join(context_parts)
        # System prompt
        system_prompt = (
            "You are an official NMIT College AI Assistant. Answer the user's question accurately based ONLY on the provided context.\n"
            "If the answer cannot be determined from the context, state clearly: 'I don't have this information in the ingested college documents.'\n"
            "Format your answer clearly with bullet points if helpful, and reference the source numbers used."
        )
        # Build final user prompt
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return self._call_hf_api(prompt=user_prompt, system_prompt=system_prompt)

    # Convenience ingestion methods
    def ingest_document(self, filepath: str) -> None:
        from document_processor import load_document, chunk_text
        text = load_document(filepath)
        source_name = os.path.basename(filepath)
        chunks = chunk_text(text, source_name=source_name)
        self.add_chunks(chunks)

    def ingest_url(self, url: str) -> None:
        from web_scraper import scrape_and_chunk
        chunks = scrape_and_chunk(url)
        self.add_chunks(chunks)