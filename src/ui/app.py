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
        return response.content
    except Exception as exc:
        traceback.print_exc()
        return (
            f"❌ **An error occurred while contacting the Gemini API.**\n\n"
            f"Details: `{exc}`\n\n"
            "Please check your API key and network connection, then try again."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global fonts & palette ───────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary:    #2563eb;
    --primary-dk: #1d4ed8;
    --accent:     #06b6d4;
    --bg:         #0f172a;
    --surface:    #1e293b;
    --surface2:   #273549;
    --text:       #e2e8f0;
    --text-muted: #94a3b8;
    --border:     #334155;
    --radius:     14px;
    --shadow:     0 4px 24px rgba(0,0,0,.45);
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Header banner ─────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0c4a6e 50%, #164e63 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 36px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 80% 50%, rgba(6,182,212,.15) 0%, transparent 60%);
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    background: linear-gradient(90deg, #93c5fd, #67e8f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
}
.app-header p {
    color: var(--text-muted);
    font-size: .95rem;
    margin: 0;
}

/* ── Chat bubbles ───────────────────────────────────────────────────────── */
.message.user   { background: var(--surface2) !important; border-radius: var(--radius) !important; }
.message.bot    { background: var(--surface)  !important; border-radius: var(--radius) !important; }

/* ── Input row ──────────────────────────────────────────────────────────── */
.input-row textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.input-row textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(6,182,212,.2) !important;
}

/* ── Submit button ──────────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    transition: filter .2s, transform .15s !important;
}
button.primary:hover {
    filter: brightness(1.12) !important;
    transform: translateY(-1px) !important;
}

/* ── Disclaimer card ────────────────────────────────────────────────────── */
.disclaimer {
    background: rgba(245,158,11,.08);
    border: 1px solid rgba(245,158,11,.35);
    border-radius: var(--radius);
    padding: 14px 20px;
    margin-top: 16px;
    font-size: .85rem;
    color: #fcd34d;
}
"""

HEADER_HTML = """
<div class="app-header">
  <h1>🦷 Dental Diagnosis &amp; Education Assistant</h1>
  <p>Multimodal RAG · Powered by ChromaDB + Gemini 1.5 Flash</p>
</div>
"""

DISCLAIMER_HTML = """
<div class="disclaimer">
  ⚠️ <strong>Educational use only.</strong> This assistant draws on a curated dental knowledge base
  and an AI language model. It does <em>not</em> replace professional dental examination or treatment.
  Always consult a licensed dentist for personal health decisions.
</div>
"""

EXAMPLE_QUERIES = [
    ["Apa saja ciri-ciri awal penyakit gusi?", []],
    ["Bagaimana cara meredakan sakit gigi di rumah?", []],
    ["Apa penyebab gigi sensitif dan cara mengatasinya?", []],
    ["Jelaskan perbedaan karang gigi dan plak.", []],
    ["Apakah aman melakukan pemutihan gigi sendiri?", []], # Tambahan baru
]


# Gradio 6.0: theme and css are passed to launch(), NOT to gr.Blocks()
_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Dental Diagnosis & Education Assistant") as demo:

        gr.HTML(HEADER_HTML)

        # Gradio 6.0: retry_btn / undo_btn / clear_btn are removed
        chat = gr.ChatInterface(  # noqa: F841
            fn=process_query,
            multimodal=True,
            chatbot=gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=False,
                render_markdown=True,
                avatar_images=(
                    None,                          # user — default
                    "https://api.dicebear.com/8.x/bottts/svg?seed=dental",
                ),
            ),
            textbox=gr.MultimodalTextbox(
                placeholder="Ask about dental health, or upload an image of a dental concern…",
                show_label=False,
                file_types=["image"],
                elem_classes=["input-row"],
            ),
            examples=EXAMPLE_QUERIES,
            cache_examples=False,
            # submit_btn removed: conflicts with custom MultimodalTextbox in Gradio 6.0
        )

        gr.HTML(DISCLAIMER_HTML)

    return demo


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    # Gradio 6.0: theme and css belong in launch(), not gr.Blocks()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=_THEME,
        css=CUSTOM_CSS,
    )
