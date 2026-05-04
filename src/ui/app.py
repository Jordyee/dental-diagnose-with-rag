"""
Dental Diagnosis & Education Assistant — Multimodal RAG Application
====================================================================
Phase 2: RAG Logic + Gradio UI

Architecture:
  User query (text + optional image)
       │
       ▼
  ChromaDB retriever  ──► Relevant dental context chunks
       │
       ▼
  Gemini 1.5 Flash (multimodal)
       │
       ▼
  Structured, medically-informed response
"""

import os
import base64
import traceback
from pathlib import Path

from dotenv import load_dotenv

# ── LangChain / Embeddings ────────────────────────────────────────────────────
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Google Generative AI ──────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ── Gradio ────────────────────────────────────────────────────────────────────
import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
# 0. Environment & Paths
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Please set it in your .env file."
    )

# Resolve paths relative to this file so the app works regardless of CWD
_THIS_DIR   = Path(__file__).resolve().parent          # src/ui/
_SRC_DIR    = _THIS_DIR.parent                         # src/
_BASE_DIR   = _SRC_DIR.parent                          # project root
CHROMA_PATH = _BASE_DIR / "data" / "chroma_db_gigi"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Retriever — Load persisted ChromaDB
# ─────────────────────────────────────────────────────────────────────────────

print(f"[INFO] Loading ChromaDB from: {CHROMA_PATH}")
_embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=_embedding_model,
)

# k=4 gives a good balance between context richness and prompt length
RETRIEVER = _vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
print("[INFO] ChromaDB retriever initialised successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM — Gemini 1.5 Flash (multimodal)
# ─────────────────────────────────────────────────────────────────────────────

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,          # slightly creative but stay factual
    max_output_tokens=2048,
)
print("[INFO] Gemini 2.5 Flash LLM initialised successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Helper — encode image file to base64 data-URI
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> tuple[str, str]:
    """
    Returns (mime_type, base64_data) for the given image file path.
    Supports JPEG, PNG, WEBP, and GIF.
    """
    suffix = Path(image_path).suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return mime_type, data


# ─────────────────────────────────────────────────────────────────────────────
# 4. Core RAG function
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert dental education and diagnosis assistant powered by a curated knowledge base of dental literature.

Your role:
• Provide detailed, medically accurate information about dental conditions, symptoms, treatments, and preventive care.
• When an image is provided, analyse it carefully for any visible dental concerns (discolouration, lesions, abnormal tissue, etc.).
• Integrate the retrieved context (from the knowledge base) with the image analysis and the user's query to give a comprehensive, well-structured answer.
• Always recommend consulting a licensed dentist for professional diagnosis and treatment.
• Use clear, accessible language — avoid excessive jargon, but do not oversimplify important details.
• Structure longer answers with short headings or bullet points for readability.

IMPORTANT DISCLAIMER: This assistant is for educational purposes only and does not replace professional dental advice."""


def process_query(message: dict, history: list) -> str:
    """
    Main RAG handler called by gr.ChatInterface (multimodal=True).

    Parameters
    ----------
    message : dict  — {"text": str, "files": list[str]}
    history : list  — conversation history (unused for retrieval, kept for UI)

    Returns
    -------
    str — The assistant's response.
    """

    # ── 4a. Extract user text & optional image ────────────────────────────
    user_text: str = (message.get("text") or "").strip()
    uploaded_files: list = message.get("files") or []

    if not user_text and not uploaded_files:
        return "⚠️ Please enter a question or upload a dental image (or both)."

    # Provide a default query for image-only submissions
    retrieval_query = user_text if user_text else "dental image analysis oral health"

    # ── 4b. Retrieve relevant context from ChromaDB ───────────────────────
    try:
        relevant_docs = RETRIEVER.invoke(retrieval_query)
        context_text = "\n\n---\n\n".join(
            [f"[Source {i+1}]\n{doc.page_content}" for i, doc in enumerate(relevant_docs)]
        )
    except Exception as exc:
        print(f"[ERROR] Retrieval failed: {exc}")
        context_text = "(Knowledge base retrieval temporarily unavailable.)"

    # ── 4c. Build the multimodal message for Gemini ───────────────────────
    #   Content is a list of parts: text first, then inline image data.
    content_parts: list = []

    # System instructions + retrieved context + user query (all as one text block)
    full_prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"## Relevant Knowledge Base Context\n\n{context_text}\n\n"
        f"## User Query\n\n{user_text if user_text else '(No text — please analyse the uploaded image.)'}"
    )
    content_parts.append({"type": "text", "text": full_prompt})

    # Attach image(s) if provided
    for file_path in uploaded_files:
        try:
            mime_type, b64_data = _encode_image(file_path)
            content_parts.append({
                "type":       "image_url",
                "image_url":  {"url": f"data:{mime_type};base64,{b64_data}"},
            })
        except Exception as exc:
            print(f"[WARN] Could not encode image '{file_path}': {exc}")
            content_parts.append({
                "type": "text",
                "text": f"(Note: An image was uploaded but could not be processed — {exc})",
            })

    # ── 4d. Call the LLM ──────────────────────────────────────────────────
    try:
        response = LLM.invoke([HumanMessage(content=content_parts)])
        
        # Append RAG Sources to the response
        final_answer = response.content
        if "relevant_docs" in locals() and relevant_docs:
            final_answer += "\n\n<details><summary>📚 Sources</summary>\n\n"
            for i, doc in enumerate(relevant_docs):
                # Clean up the page_content slightly for display
                clean_text = doc.page_content.replace('\n', ' ').strip()
                final_answer += f"**Source {i+1}**:\n> {clean_text}\n\n"
            final_answer += "</details>"
            
        return final_answer
    except Exception as exc:
        traceback.print_exc()
        return (
            f"❌ **An error occurred while contacting the Gemini API.**\n\n"
            f"Details: `{exc}`\n\n"
            "Please check your API key and network connection, then try again."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gradio UI — Redesigned Two-Column Layout
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #273549;
    --bg-hover: #2d3f56;
    --accent: #0ea5e9;
    --accent-teal: #14b8a6;
    --accent-glow: rgba(14,165,233,.15);
    --teal-glow: rgba(20,184,166,.12);
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #94a3b8;
    --border: #334155;
    --border-light: #475569;
    --warning: #f59e0b;
    --warning-bg: rgba(245,158,11,.07);
    --success: #22c55e;
    --radius: 14px;
    --radius-sm: 10px;
    --radius-xs: 6px;
    --shadow: 0 4px 24px rgba(0,0,0,.4);
    --shadow-sm: 0 2px 8px rgba(0,0,0,.3);
}

/* ── Global ──────────────────────────────────────────────────────────── */
body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.gradio-container { max-width: 1400px !important; }

/* ── Header ──────────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0c2d48 0%, #0c4a6e 40%, #134e5e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px 32px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 85% 50%, rgba(14,165,233,.12) 0%, transparent 55%);
    pointer-events: none;
}
.header-content { position: relative; z-index: 1; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; }
.header-left h1 {
    font-size: 1.6rem; font-weight: 700; margin: 0 0 4px;
    background: linear-gradient(90deg, #93c5fd, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header-left p { color: var(--text-muted); font-size: .88rem; margin: 0; }
.status-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,.06); border: 1px solid var(--border);
    border-radius: 20px; padding: 5px 14px; font-size: .75rem;
    color: var(--text-secondary); font-weight: 500; white-space: nowrap;
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
.status-dot.green { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-dot.blue  { background: var(--accent);  box-shadow: 0 0 6px var(--accent);  }
@keyframes pulse-dot {
    0%, 100% { opacity: 1; } 50% { opacity: .5; }
}

/* ── Sidebar ─────────────────────────────────────────────────────────── */
.sidebar-panel {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
}
.sidebar-panel h3 {
    font-size: .95rem; font-weight: 600; color: var(--text-primary);
    margin: 0 0 12px; display: flex; align-items: center; gap: 8px;
}
.sidebar-panel p, .sidebar-panel li {
    font-size: .84rem; color: var(--text-secondary); line-height: 1.6; margin: 0;
}
.sidebar-panel ul { padding-left: 18px; margin: 8px 0 0; }
.sidebar-panel li { margin-bottom: 4px; }

/* ── Chip buttons ────────────────────────────────────────────────────── */
.chip-btn {
    width: 100%;
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-size: .82rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    padding: 10px 14px !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: all .2s ease !important;
    margin-bottom: 6px !important;
    min-height: unset !important;
    line-height: 1.4 !important;
}
.chip-btn:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
    color: var(--text-primary) !important;
    transform: translateX(3px);
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}
.chip-btn:active { transform: translateX(1px) scale(.99); }

/* ── Disclaimer ──────────────────────────────────────────────────────── */
.disclaimer-box {
    background: var(--warning-bg);
    border: 1px solid rgba(245,158,11,.3);
    border-radius: var(--radius-sm);
    padding: 14px 16px;
    font-size: .8rem;
    color: #fcd34d;
    line-height: 1.55;
}
.disclaimer-box strong { color: #fbbf24; }

/* ── Chat area ───────────────────────────────────────────────────────── */
.chat-column {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0;
    overflow: hidden;
    box-shadow: var(--shadow);
}

/* Chatbot window */
.chat-window {
    border: none !important;
    background: transparent !important;
}
.chat-window .message-row {
    padding: 6px 16px !important;
}
.chat-window .bot, .chat-window .message.bot {
    background: rgba(14,165,233,.06) !important;
    border: 1px solid rgba(14,165,233,.12) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    line-height: 1.65 !important;
    font-size: .9rem !important;
}
.chat-window .user, .chat-window .message.user {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    line-height: 1.65 !important;
    font-size: .9rem !important;
}

/* Avatar styling */
.chat-window .avatar-container img {
    border-radius: 50% !important;
    border: 2px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* Markdown inside chat */
.chat-window .message p { margin: 4px 0; }
.chat-window .message ul, .chat-window .message ol { padding-left: 20px; margin: 6px 0; }
.chat-window .message h1, .chat-window .message h2, .chat-window .message h3 {
    color: var(--accent) !important;
    margin: 12px 0 6px;
    font-weight: 600;
}
.chat-window .message code {
    background: rgba(0,0,0,.3); padding: 2px 6px; border-radius: 4px; font-size: .85em;
}
.chat-window .message table { border-collapse: collapse; width: 100%; margin: 8px 0; }
.chat-window .message th, .chat-window .message td {
    border: 1px solid var(--border); padding: 6px 10px; font-size: .84rem;
}
.chat-window .message th { background: var(--bg-tertiary); font-weight: 600; }

/* ── Input area ──────────────────────────────────────────────────────── */
.chat-input {
    border-top: 1px solid var(--border) !important;
    background: var(--bg-secondary) !important;
    padding: 8px !important;
}
.chat-input textarea {
    background: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .9rem !important;
    line-height: 1.5 !important;
    padding: 12px 16px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.chat-input textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
.chat-input button[aria-label] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    transition: all .2s !important;
}
.chat-input button[aria-label]:hover {
    border-color: var(--accent) !important;
    background: var(--accent-glow) !important;
}

/* Submit button in input area */
.chat-input button.primary, .chat-input button[class*="submit"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-teal)) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    color: white !important;
    transition: filter .2s, transform .15s !important;
    box-shadow: 0 2px 8px rgba(14,165,233,.3) !important;
}
.chat-input button.primary:hover, .chat-input button[class*="submit"]:hover {
    filter: brightness(1.15) !important;
    transform: translateY(-1px) !important;
}

/* ── Chat Actions ────────────────────────────────────────────────────── */
.chat-actions {
    background: var(--bg-secondary);
    padding: 8px 16px 16px 16px; 
    border-bottom-left-radius: var(--radius);
    border-bottom-right-radius: var(--radius);
}
.action-btn {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}
.action-btn:hover {
    background: var(--bg-hover) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-light) !important;
}

/* ── Capabilities list ───────────────────────────────────────────────── */
.capability-tag {
    display: inline-block;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: .74rem;
    color: var(--text-muted);
    margin: 3px 2px;
}

/* ── Responsive ──────────────────────────────────────────────────────── */
@media (max-width: 860px) {
    .main-row { flex-direction: column !important; }
    .main-row > .gr-column { min-width: 100% !important; }
}

/* ── Loading / thinking animation ────────────────────────────────────── */
@keyframes thinking-pulse {
    0%, 100% { opacity: .6; }
    50% { opacity: 1; }
}
.thinking-indicator {
    animation: thinking-pulse 1.5s ease-in-out infinite;
    color: var(--accent) !important;
    font-style: italic;
}

/* ── Scrollbar ───────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

/* ── Footer ──────────────────────────────────────────────────────────── */
footer { opacity: .4 !important; }
"""

# ── HTML fragments ────────────────────────────────────────────────────────────

HEADER_HTML = """
<div class="app-header">
  <div class="header-content">
    <div class="header-left">
      <h1>🦷 Dental Health AI Assistant</h1>
      <p>Multimodal RAG · Powered by ChromaDB + Gemini 2.5 Flash</p>
    </div>
    <div class="status-badges">
      <span class="status-badge"><span class="status-dot green"></span>Local RAG Active</span>
      <span class="status-badge"><span class="status-dot blue"></span>Gemini Vision Ready</span>
    </div>
  </div>
</div>
"""

INFO_HTML = """
<div class="sidebar-panel">
  <h3>💡 How It Works</h3>
  <p>Upload a photo of your teeth or describe your symptoms, and the AI will:</p>
  <ul>
    <li>Search a curated dental knowledge base</li>
    <li>Analyse uploaded images for visible concerns</li>
    <li>Provide detailed, evidence-backed explanations</li>
  </ul>
  <p style="margin-top:10px; font-size:.78rem; color:var(--text-muted);">
    Supports <strong>JPG, PNG, WEBP, GIF</strong> images.
  </p>
</div>
"""

CAPABILITIES_HTML = """
<div class="sidebar-panel">
  <h3>📋 Knowledge Areas</h3>
  <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">
    <span class="capability-tag">Dental Caries</span>
    <span class="capability-tag">Gum Disease</span>
    <span class="capability-tag">Gingivitis</span>
    <span class="capability-tag">Periodontitis</span>
    <span class="capability-tag">Oral Hygiene</span>
    <span class="capability-tag">AI Dental Imaging</span>
  </div>
</div>
"""

DISCLAIMER_HTML = """
<div class="sidebar-panel" style="background:var(--warning-bg); border-color:rgba(245,158,11,.3);">
  <div class="disclaimer-box" style="background:transparent; border:none; padding:0;">
    ⚠️ <strong>Educational use only.</strong> This assistant does <em>not</em> replace
    professional dental examination or treatment. Always consult a licensed dentist
    for personal health decisions.
  </div>
</div>
"""

PRESET_QUERIES = [
    ("🦷 Early signs of gum disease?",  "What are the early signs of gum disease?"),
    ("🔍 Gingivitis vs Periodontitis",  "Explain the difference between gingivitis and periodontitis."),
    ("🏠 Treat sensitivity at home",    "How can I treat tooth sensitivity at home?"),
    ("🛡️ Preventing tooth decay",       "What causes tooth decay and how is it prevented?"),
]


# ── Theme ─────────────────────────────────────────────────────────────────────

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
)


# ── Chat event handlers ──────────────────────────────────────────────────────

def _add_user_message(message, history):
    """Instantly show user message in chat and clear the input box."""
    user_text = (message.get("text") or "").strip()
    user_files = message.get("files") or []

    if not user_text and not user_files:
        return history, message, gr.update()

    # Display uploaded images
    for f in user_files:
        fp = f["path"] if isinstance(f, dict) else f
        history.append({"role": "user", "content": {"path": fp}})

    # Display text
    if user_text:
        history.append({"role": "user", "content": user_text})

    return history, message, gr.MultimodalTextbox(value=None)


def _generate_response(saved_msg, history):
    """Call the RAG pipeline and stream the response into chat."""
    if saved_msg is None:
        yield history
        return

    # Show thinking indicator
    history.append({
        "role": "assistant",
        "content": "⏳ *Analyzing your query and searching the dental knowledge base…*"
    })
    yield history

    # Get the real response
    response = process_query(saved_msg, history)
    history[-1] = {"role": "assistant", "content": response}
    yield history


def _make_chip_handler(query_text):
    """Create a handler for a preset chip button."""
    def handler(history):
        history.append({"role": "user", "content": query_text})
        history.append({
            "role": "assistant",
            "content": "⏳ *Searching dental knowledge base…*"
        })
        yield history

        message = {"text": query_text, "files": []}
        response = process_query(message, history)
        history[-1] = {"role": "assistant", "content": response}
        yield history
    return handler


def handle_like(data: gr.LikeData):
    """Handle feedback on AI responses."""
    print(f"[Feedback] {'👍' if data.liked else '👎'} on message: {data.value}")

def export_chat(history):
    """Export the chat history as a formatted TXT file."""
    import tempfile
    content = "Dental Health AI Assistant - Chat History\n"
    content += "="*40 + "\n\n"
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        text = msg["content"]
        if isinstance(text, dict) and "path" in text:
            text = f"[Image Attached: {text['path']}]"
        elif isinstance(text, tuple):
            text = f"[File Attached: {text[0]}]"
        content += f"{role}:\n{text}\n\n"
        content += "-"*40 + "\n\n"
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
        f.write(content)
        return gr.update(value=f.name, visible=True)


# ── Build the UI ──────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Dental Health AI Assistant") as demo:

        # State to hold the raw message dict between event steps
        pending_msg = gr.State(None)

        # ── Header
        gr.HTML(HEADER_HTML)

        # ── Main two-column layout
        with gr.Row(elem_classes=["main-row"]):

            # ── LEFT: Sidebar (30%)
            with gr.Column(scale=3, min_width=260):
                gr.HTML(INFO_HTML)

                gr.HTML('<div class="sidebar-panel"><h3>⚡ Quick Start</h3></div>'
                        .replace('</div>', ''))
                chips = []
                for label, _query in PRESET_QUERIES:
                    btn = gr.Button(label, elem_classes=["chip-btn"], variant="secondary")
                    chips.append((btn, _query))
                gr.HTML('</div>')  # close the sidebar-panel

                gr.HTML(CAPABILITIES_HTML)
                gr.HTML(DISCLAIMER_HTML)

            # ── RIGHT: Chat (70%)
            with gr.Column(scale=7, min_width=400, elem_classes=["chat-column"]):
                gr.HTML("""<div style="background:var(--warning-bg); padding:8px 16px; font-size:.85rem; color:#fcd34d; border-bottom:1px solid rgba(245,158,11,.3); text-align:center;">
                  ⚠️ <strong>Disclaimer:</strong> This AI is for educational purposes only and draws from a curated knowledge base. It does not replace professional dental examination or treatment. Always consult a licensed dentist.
                </div>""")
                
                chatbot = gr.Chatbot(
                    height=540,
                    show_label=False,
                    render_markdown=True,
                    elem_classes=["chat-window"],
                    placeholder="Ask a question or upload a dental image to get started…",
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/8.x/bottts/svg?seed=dental",
                    ),
                    layout="bubble",
                )

                msg = gr.MultimodalTextbox(
                    placeholder="Describe your dental concern or upload an image…",
                    show_label=False,
                    file_types=["image"],
                    elem_classes=["chat-input"],
                    submit_btn=True,
                    stop_btn=False,
                )
                
                with gr.Row(elem_classes=["chat-actions"]):
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm", elem_classes=["action-btn"])
                    export_btn = gr.Button("📄 Export as TXT", variant="secondary", size="sm", elem_classes=["action-btn"])
                export_file = gr.File(visible=False, label="Download Chat History")

        # ── Wire events: submit via textbox
        msg.submit(
            _add_user_message, [msg, chatbot], [chatbot, pending_msg, msg]
        ).then(
            _generate_response, [pending_msg, chatbot], [chatbot]
        )

        # ── Wire events: preset chip buttons
        for btn, query_text in chips:
            btn.click(
                _make_chip_handler(query_text), [chatbot], [chatbot]
            )
            
        clear_btn.click(lambda: [], None, chatbot)
        export_btn.click(export_chat, [chatbot], [export_file])
        chatbot.like(handle_like, None, None)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=_THEME,
        css=CUSTOM_CSS,
    )
