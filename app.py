import os
# Force CPU mode and prevent HF Spaces ZeroGPU CUDA conflict
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    import spaces
except Exception:
    pass

import gradio as gr

# Load .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system env vars

from rag_engine import RagEngine

# Initialize RAG engine (will read HF_TOKEN from environment)
rag_engine = RagEngine()

def ingest_file(file):
    """
    Handle uploaded file: save temporarily, ingest into RAG engine.
    Returns status message.
    """
    if file is None:
        return "No file uploaded."
    try:
        # file is a tuple (filename, file_data) or similar depending on Gradio version
        # In Gradio 4, file is a NamedString or similar with .name attribute
        filepath = file.name if hasattr(file, 'name') else file
        rag_engine.ingest_document(filepath)
        return f"Successfully ingested: {os.path.basename(filepath)}"
    except Exception as e:
        return f"Error ingesting file: {e}"

def ingest_url(url):
    """
    Handle URL input: scrape and ingest.
    """
    if not url or not url.strip():
        return "Please enter a URL."
    try:
        rag_engine.ingest_url(url.strip())
        return f"Successfully ingested URL: {url}"
    except Exception as e:
        return f"Error ingesting URL: {e}"

def chat_function(message, history):
    """
    Gradio chatbot function: takes user message and history, returns response.
    history is a list of tuples (user_msg, bot_msg).
    """
    try:
        # Retrieve relevant chunks
        chunks = rag_engine.query(message, top_k=3)
        if not chunks:
            return "I don't have enough information to answer that question. Please upload some documents or scrape a URL first."
        # Generate answer
        answer = rag_engine.generate_answer(message, chunks)
        # Format sources for display
        sources_info = "\n\n**Sources:**\n"
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'unknown')
            # Truncate long source names
            if len(source) > 50:
                source = source[:47] + "..."
            sources_info += f"{i}. {source} (similarity: {chunk.get('score', 'N/A'):.4f})\n"
        return answer + sources_info
    except Exception as e:
        return f"Error: {e}"

def clear_chat():
    return None, None  # Clear chatbot and reset state if needed

# Build Gradio interface
with gr.Blocks(title="College RAG Chatbot") as demo:
    gr.Markdown("# 🎓 College RAG Chatbot")
    gr.Markdown("Upload college documents (PDF, DOCX, TXT) or scrape public college websites to build a knowledge base, then ask questions!")

    with gr.Tab("Ingest Data"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload Document (PDF, DOCX, TXT)", file_types=[".pdf", ".docx", ".txt"])
                ingest_file_btn = gr.Button("Ingest Uploaded File")
                file_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                url_input = gr.Textbox(label="Enter College Website URL", placeholder="https://example.edu/info")
                ingest_url_btn = gr.Button("Ingest URL")
                url_status = gr.Textbox(label="Status", interactive=False)

        ingest_file_btn.click(ingest_file, inputs=file_input, outputs=file_status)
        ingest_url_btn.click(ingest_url, inputs=url_input, outputs=url_status)

    with gr.Tab("Chat"):
        gr.Markdown("### Ask questions about your college data")
        chatbot = gr.Chatbot(label="Conversation", height=400)
        with gr.Row():
            msg = gr.Textbox(label="Your Question", placeholder="Ask about courses, placements, facilities...", lines=2, scale=4)
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
        gr.Markdown(
            "**Try asking:** *What courses are offered?* · *What are the placement stats?* "
            "· *Tell me about hostel facilities* · *What is the admission process?*"
        )

        # Handle submission
        def respond(message, chat_history):
            if not message or not message.strip():
                return "", chat_history
            bot_message = chat_function(message, chat_history)
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": bot_message})
            return "", chat_history

        submit_btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear_btn.click(lambda: [], None, chatbot, queue=False)

    with gr.Tab("About"):
        gr.Markdown("""
        ## How it works
        1. **Ingest**: Upload documents or provide URLs to college websites.
        2. **Process**: Text is extracted, chunked, and embedded using sentence-transformers.
        3. **Store**: Chunks are saved in ChromaDB vector database.
        4. **Query**: When you ask a question, the system retrieves relevant chunks.
        5. **Answer**: The retrieved context is sent to a Hugging Face LLM (Mistral-7B-Instruct) to generate a grounded answer.
        6. **Cite**: The answer is based solely on the uploaded/scraped content.

        **Tech Stack**
        - LLM: Hugging Face Inference API (Mistral-7B-Instruct-v0.2)
        - Embeddings: sentence-transformers/all-MiniLM-L6-v2
        - Vector Store: ChromaDB (local persistence)
        - UI: Gradio
        - File Parsing: PyPDF2, python-docx
        - Web Scraping: BeautifulSoup4, requests

        **Deployment**
        - Built for Hugging Face Spaces (Gradio SDK)
        - Zero cost: all components have free tiers or are open-source.
        """)

if __name__ == "__main__":
    demo.launch()