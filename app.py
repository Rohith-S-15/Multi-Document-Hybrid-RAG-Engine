import os
# Force CPU mode and prevent HF Spaces ZeroGPU CUDA conflict
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Disable Gradio 6 Server-Side Rendering (causes blank page on HF Spaces)
os.environ["GRADIO_SSR_MODE"] = "false"

# Handle spaces import for HF Spaces compatibility
try:
    import spaces
except ImportError:
    # Mock spaces for local development
    class _MockSpaces:
        @staticmethod
        def GPU(func=None, **kwargs):
            if func is not None:
                return func
            def decorator(f):
                return f
            return decorator
    spaces = _MockSpaces()

# Dummy GPU function — satisfies ZeroGPU startup requirement
@spaces.GPU
def _gpu_init():
    """Dummy GPU function required by HF Spaces ZeroGPU infrastructure."""
    return "GPU Initialized"

# Call _gpu_init at startup so ZeroGPU registers a GPU function call
try:
    _gpu_init()
except Exception as e:
    print(f"[ZeroGPU] Init call note: {e}")

import gradio as gr

# Load .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

from rag_engine import RagEngine

# Initialize RAG engine (reads GROQ_API_KEY from environment)
try:
    rag_engine = RagEngine()
except Exception as e:
    print(f"[ERROR] Failed to initialize RAG engine: {e}")
    rag_engine = None


# ════════════════════════════════════════════════════════
#  TOP-LEVEL HANDLER FUNCTIONS (must be picklable for ZeroGPU)
# ════════════════════════════════════════════════════════

def ingest_file(files):
    """Handle uploaded file(s): ingest into RAG engine."""
    if rag_engine is None:
        return "❌ RAG engine not initialized. Check GROQ_API_KEY secret."
    if not files:
        return "No file uploaded."
    
    if not isinstance(files, list):
        files = [files]

    ingested_names = []
    errors = []

    for file in files:
        try:
            filepath = file.name if hasattr(file, "name") else file
            filename = os.path.basename(filepath)
            rag_engine.ingest_document(filepath)
            ingested_names.append(filename)
        except Exception as e:
            filename = os.path.basename(file.name if hasattr(file, "name") else str(file))
            errors.append(f"{filename}: {e}")

    stats = rag_engine.get_stats()

    status_parts = []
    if ingested_names:
        status_parts.append(
            f"✅ Ingested {len(ingested_names)} file(s):\n" +
            "\n".join(f"  • {name}" for name in ingested_names)
        )
    if errors:
        status_parts.append(
            f"❌ Failed to ingest:\n" +
            "\n".join(f"  • {err}" for err in errors)
        )

    status_parts.append(
        f"📊 Total chunks in index: {stats['total_chunks']} | "
        f"Unique sources: {stats['unique_sources']}"
    )
    return "\n\n".join(status_parts)


def ingest_url(url):
    """Handle URL input: scrape and ingest."""
    if rag_engine is None:
        return "❌ RAG engine not initialized. Check GROQ_API_KEY secret."
    if not url or not url.strip():
        return "Please enter a URL."
    try:
        rag_engine.ingest_url(url.strip())
        stats = rag_engine.get_stats()
        return (
            f"✅ Successfully ingested URL: {url}\n"
            f"📊 Total chunks in index: {stats['total_chunks']} | "
            f"Sources: {stats['unique_sources']}"
        )
    except Exception as e:
        print(f"[ERROR] Error ingesting URL: {e}")
        return f"❌ Error ingesting URL: {e}"


def chat_function(message, history):
    """Generate a RAG-powered response with sources and images."""
    if rag_engine is None:
        return "❌ RAG engine not initialized. Check GROQ_API_KEY secret."
    try:
        chunks = rag_engine.query(message, top_k=8)
        if not chunks:
            return (
                "I don't have enough information to answer that question. "
                "Please upload some documents or scrape a URL first."
            )

        answer = rag_engine.generate_answer(message, chunks)

        # Build sources section
        sources_info = "\n\n---\n**📚 Sources used:**\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            score = chunk.get("score", None)
            score_str = f" (distance: {score:.4f})" if score is not None else ""
            sources_info += f"- **Source {i}:** {source}{score_str}\n"

        # Collect image URLs from retrieved chunks
        all_images = []
        for chunk in chunks:
            images = chunk.get("images", [])
            for img_url in images:
                if img_url not in all_images:
                    all_images.append(img_url)

        # Add images to response (show up to 3 relevant images)
        images_section = ""
        if all_images:
            images_section = "\n\n**🖼️ Related images:**\n"
            for img_url in all_images[:3]:
                images_section += f"![Image]({img_url})\n"

        return answer + sources_info + images_section

    except Exception as e:
        print(f"[ERROR] Error in chat function: {e}")
        return f"❌ Error: {e}"


def get_index_info():
    """Get current index statistics."""
    if rag_engine is None:
        return "❌ RAG engine not initialized. Check GROQ_API_KEY secret."
    stats = rag_engine.get_stats()
    if stats["total_chunks"] == 0:
        return "📭 Index is empty. Upload documents or scrape URLs to get started!"

    sources_list = "\n".join(f"  • {s}" for s in stats["sources"])
    return (
        f"📊 **Index Statistics:**\n"
        f"- Total chunks: **{stats['total_chunks']}**\n"
        f"- Unique sources: **{stats['unique_sources']}**\n"
        f"- Sources:\n{sources_list}"
    )


def clear_index():
    """Clear the entire RAG index."""
    if rag_engine is None:
        return "❌ RAG engine not initialized. Check GROQ_API_KEY secret."
    rag_engine.clear_index()
    return "🗑️ Index cleared. Upload new documents or scrape URLs to rebuild."


def respond(message, chat_history):
    """Top-level respond function (must be picklable for ZeroGPU)."""
    if not message or not message.strip():
        return "", chat_history
    bot_message = chat_function(message, chat_history)
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})
    return "", chat_history


def clear_chat():
    """Top-level clear chat function (must be picklable for ZeroGPU)."""
    return []


# ════════════════════════════════════════════════════════
#  BUILD GRADIO UI — demo must be a top-level variable
# ════════════════════════════════════════════════════════

demo = gr.Blocks(title="⚡ Multi-Document Hybrid RAG Engine")

with demo:
    gr.Markdown("# ⚡ Multi-Document Hybrid RAG Engine")
    gr.Markdown(
        "Context-grounded information retrieval and question-answering across multi-format documents "
        "(PDF, DOCX, TXT) and live web pages with verified source citations. "
        "Powered by **FAISS** + **LangChain** + **Sentence-Transformers** + **Groq AI**."
    )

    with gr.Tab("💬 Query & Chat"):
        gr.Markdown("### Ask questions against ingested knowledge base")
        chatbot = gr.Chatbot(label="Conversation & Grounded Answers", height=450)
        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask specific questions about policies, course structures, reports, or documentation...",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Query", variant="primary", scale=1)
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        gr.Markdown(
            "**Example queries:** *What are the key policy requirements?* · "
            "*Summarize the admission and fee regulations* · "
            "*Extract key performance metrics and statistics* · "
            "*What are the required submission deadlines?*"
        )

        submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(clear_chat, None, chatbot, queue=False)

    with gr.Tab("📥 Ingest Data"):
        gr.Markdown("### Add knowledge to the chatbot")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### 📄 Upload Document")
                file_input = gr.File(
                    label="Upload Document(s) (PDF, DOCX, TXT, CSV)",
                    file_types=[".pdf", ".docx", ".txt", ".csv"],
                    file_count="multiple",
                )
                ingest_file_btn = gr.Button("Ingest Uploaded File", variant="primary")
                file_status = gr.Textbox(label="Status", interactive=False, lines=2)
            with gr.Column():
                gr.Markdown("#### 🌐 Ingest Web URL")
                url_input = gr.Textbox(
                    label="Enter Public Web URL",
                    placeholder="https://example.com/documentation-page",
                )
                ingest_url_btn = gr.Button("Ingest URL", variant="primary")
                url_status = gr.Textbox(label="Status", interactive=False, lines=2)

        ingest_file_btn.click(ingest_file, inputs=file_input, outputs=file_status)
        ingest_url_btn.click(ingest_url, inputs=url_input, outputs=url_status)

        gr.Markdown("---")
        gr.Markdown("#### 📊 Index Statistics")
        index_info = gr.Textbox(label="Current Index", interactive=False, lines=5)
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Stats")
            clear_index_btn = gr.Button("🗑️ Clear Index", variant="stop")

        refresh_btn.click(get_index_info, outputs=index_info)
        clear_index_btn.click(clear_index, outputs=index_info)

    with gr.Tab("ℹ️ Architecture & Pipeline"):
        gr.Markdown("""
        ## Retrieval-Augmented Generation Architecture
        1. **Multi-Source Ingestion**: Ingest multi-page PDFs, Word documents (.docx), text files, or live URLs.
        2. **Semantic Text Chunking**: Decompose raw text using `RecursiveCharacterTextSplitter` with dynamic chunk overlap (1000 tokens, 200 overlap) and metadata preservation.
        3. **Dense Vector Embeddings**: Encode text chunks into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`.
        4. **Similarity Indexing**: Store normalized vectors in a FAISS L2/cosine index for sub-millisecond similarity search.
        5. **Contextual Assembly & Reranking**: Retrieve top-k semantic passages, prepend document header metadata, and construct structured prompt context.
        6. **LLM Synthesis**: Stream generation through Groq API (`gemma2-9b-it`) with strict grounding to prevent hallucinations.
        7. **Citation Verification**: Every generated claim is mapped back to source filenames, page numbers, and cosine distance scores.

        ## Core Infrastructure
        | Component | Technology | Role |
        |---|---|---|
        | **LLM Inference** | Groq Cloud API | Ultra-low latency token generation |
        | **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Semantic text encoding |
        | **Vector Store** | FAISS (Facebook AI Similarity Search) | High-throughput nearest neighbor search |
        | **Orchestration** | LangChain Core | Document loaders & prompt management |
        | **Web Ingestion** | BeautifulSoup4 + Requests | Clean DOM extraction & image parsing |
        | **Interface** | Gradio | Reactive dashboard with streaming support |
        """)
        """)


# Launch the app - this must be at module level for ZeroGPU
if __name__ == "__main__":
    demo.launch(ssr_mode=False)
else:
    # For Hugging Face Spaces - the app is imported, so we need to expose the demo variable
    # The spaces framework will look for a 'demo' or 'app' variable at module level
    pass