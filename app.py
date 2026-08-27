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

import gradio as gr

# Load .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

from rag_engine import RagEngine

# Initialize RAG engine (reads GROQ_API_KEY from environment)
rag_engine = RagEngine()


# Dummy GPU function — satisfies ZeroGPU startup requirement
@spaces.GPU
def _gpu_init():
    return True


# ════════════════════════════════════════════════════════
#  TOP-LEVEL HANDLER FUNCTIONS (must be picklable)
# ════════════════════════════════════════════════════════

def ingest_file(file):
    """Handle uploaded file: ingest into RAG engine."""
    if file is None:
        return "No file uploaded."
    try:
        filepath = file.name if hasattr(file, "name") else file
        rag_engine.ingest_document(filepath)
        stats = rag_engine.get_stats()
        return (
            f"✅ Successfully ingested: {os.path.basename(filepath)}\n"
            f"📊 Total chunks in index: {stats['total_chunks']} | "
            f"Sources: {stats['unique_sources']}"
        )
    except Exception as e:
        return f"❌ Error ingesting file: {e}"


def ingest_url(url):
    """Handle URL input: scrape and ingest."""
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
        return f"❌ Error ingesting URL: {e}"


def chat_function(message, history):
    """Generate a RAG-powered response with sources and images."""
    try:
        chunks = rag_engine.query(message, top_k=3)
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
        return f"❌ Error: {e}"


def get_index_info():
    """Get current index statistics."""
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

demo = gr.Blocks(title="🎓 NMIT College RAG Chatbot")

with demo:
    gr.Markdown("# 🎓 NMIT College RAG Chatbot")
    gr.Markdown(
        "Upload college documents (PDF, DOCX, TXT, CSV) or scrape public college "
        "websites to build a knowledge base, then ask questions! "
        "Powered by **Groq AI** + **FAISS** + **LangChain**."
    )

    with gr.Tab("💬 Chat"):
        gr.Markdown("### Ask questions about your college data")
        chatbot = gr.Chatbot(label="Conversation", height=450)
        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask about courses, placements, facilities, admissions...",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        gr.Markdown(
            "**Try asking:** *What courses does NMIT offer?* · "
            "*What are the placement statistics?* · "
            "*Tell me about hostel facilities* · "
            "*What is the admission process?*"
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
                    label="Upload Document (PDF, DOCX, TXT, CSV)",
                    file_types=[".pdf", ".docx", ".txt", ".csv"],
                )
                ingest_file_btn = gr.Button("Ingest Uploaded File", variant="primary")
                file_status = gr.Textbox(label="Status", interactive=False, lines=2)
            with gr.Column():
                gr.Markdown("#### 🌐 Scrape Website")
                url_input = gr.Textbox(
                    label="Enter College Website URL",
                    placeholder="https://nitte.edu.in/nmit/about-nmit-bangalore.php",
                )
                ingest_url_btn = gr.Button("Ingest URL", variant="primary")
                url_status = gr.Textbox(label="Status", interactive=False, lines=2)

        ingest_file_btn.click(ingest_file, inputs=file_input, outputs=file_status)
        ingest_url_btn.click(ingest_url, inputs=url_input, outputs=url_status)

        gr.Markdown("---")
        gr.Markdown("#### 📊 Index Status")
        index_info = gr.Textbox(label="Current Index", interactive=False, lines=5)
        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Stats")
            clear_index_btn = gr.Button("🗑️ Clear Index", variant="stop")

        refresh_btn.click(get_index_info, outputs=index_info)
        clear_index_btn.click(clear_index, outputs=index_info)

    with gr.Tab("ℹ️ About"):
        gr.Markdown("""
        ## How it works
        1. **Ingest**: Upload documents or provide URLs to college websites.
        2. **Process**: Text is extracted and chunked using LangChain's RecursiveCharacterTextSplitter.
        3. **Embed**: Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2`.
        4. **Store**: Embeddings are stored in a FAISS vector index for fast similarity search.
        5. **Query**: When you ask a question, the system retrieves the most relevant chunks.
        6. **Answer**: The retrieved context is sent to **Groq AI** (qwen/qwen3.6-27b) to generate a grounded answer.
        7. **Cite**: The answer references specific sources and may include related images.

        ## Tech Stack
        | Component | Technology |
        |-----------|------------|
        | **LLM** | Groq API (qwen/qwen3.6-27b) |
        | **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
        | **Vector Store** | FAISS (Facebook AI Similarity Search) |
        | **Chunking** | LangChain RecursiveCharacterTextSplitter |
        | **Document Parsing** | LangChain loaders (PDF, DOCX, TXT, CSV) |
        | **Web Scraping** | BeautifulSoup4 + requests |
        | **UI** | Gradio |

        ## Built by
        Rohit S — NMIT Bangalore | Resume Portfolio Project
        """)


demo.launch(ssr_mode=False)