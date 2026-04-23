"""Standalone UI preview — tests only the Gradio layout + CSS, no backend."""
import gradio as gr

def process_query(message, history):
    """Mock backend for UI testing."""
    import time; time.sleep(1)
    text = (message.get("text") or "").strip()
    files = message.get("files") or []
    parts = ["## Mock Response\n"]
    if text:
        parts.append(f"You asked: *{text}*\n")
    if files:
        parts.append(f"📸 Received {len(files)} image(s) for analysis.\n")
    parts.append("- This is a **UI preview** with a mock backend.\n- The real RAG pipeline is not loaded.\n")
    parts.append("> ⚠️ Always consult a licensed dentist for personal health decisions.")
    return "\n".join(parts)

# ─── CSS ──────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --bg-primary: #0f172a; --bg-secondary: #1e293b; --bg-tertiary: #273549;
    --bg-hover: #2d3f56; --accent: #0ea5e9; --accent-teal: #14b8a6;
    --accent-glow: rgba(14,165,233,.15); --text-primary: #f8fafc;
    --text-secondary: #cbd5e1; --text-muted: #94a3b8; --border: #334155;
    --border-light: #475569; --warning: #f59e0b; --warning-bg: rgba(245,158,11,.07);
    --success: #22c55e; --radius: 14px; --radius-sm: 10px;
    --shadow: 0 4px 24px rgba(0,0,0,.4); --shadow-sm: 0 2px 8px rgba(0,0,0,.3);
}
body, .gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: var(--bg-primary) !important; color: var(--text-primary) !important;
}
.gradio-container { max-width: 1400px !important; }
.app-header {
    background: linear-gradient(135deg, #0c2d48 0%, #0c4a6e 40%, #134e5e 100%);
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 24px 32px; margin-bottom: 16px; box-shadow: var(--shadow);
    position: relative; overflow: hidden;
}
.app-header::before {
    content: ''; position: absolute; inset: 0;
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
    color: var(--text-secondary); font-weight: 500;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; animation: pulse-dot 2s ease-in-out infinite; }
.status-dot.green { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-dot.blue  { background: var(--accent);  box-shadow: 0 0 6px var(--accent);  }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: .5; } }
.sidebar-panel {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; margin-bottom: 12px; box-shadow: var(--shadow-sm);
}
.sidebar-panel h3 { font-size: .95rem; font-weight: 600; color: var(--text-primary); margin: 0 0 12px; }
.sidebar-panel p, .sidebar-panel li { font-size: .84rem; color: var(--text-secondary); line-height: 1.6; margin: 0; }
.sidebar-panel ul { padding-left: 18px; margin: 8px 0 0; }
.sidebar-panel li { margin-bottom: 4px; }
.chip-btn {
    width: 100%; background: var(--bg-tertiary) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important; color: var(--text-secondary) !important;
    font-size: .82rem !important; font-weight: 500 !important; font-family: 'Inter', sans-serif !important;
    padding: 10px 14px !important; text-align: left !important; cursor: pointer !important;
    transition: all .2s ease !important; margin-bottom: 6px !important;
    min-height: unset !important; line-height: 1.4 !important;
}
.chip-btn:hover {
    background: var(--bg-hover) !important; border-color: var(--accent) !important;
    color: var(--text-primary) !important; transform: translateX(3px);
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}
.disclaimer-box { background: var(--warning-bg); border: 1px solid rgba(245,158,11,.3);
    border-radius: var(--radius-sm); padding: 14px 16px; font-size: .8rem; color: #fcd34d; line-height: 1.55; }
.disclaimer-box strong { color: #fbbf24; }
.chat-column { background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 0; overflow: hidden; box-shadow: var(--shadow); }
.chat-window { border: none !important; background: transparent !important; }
.chat-window .bot, .chat-window .message.bot {
    background: rgba(14,165,233,.06) !important; border: 1px solid rgba(14,165,233,.12) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
    line-height: 1.65 !important; font-size: .9rem !important;
}
.chat-window .user, .chat-window .message.user {
    background: var(--bg-tertiary) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
    line-height: 1.65 !important; font-size: .9rem !important;
}
.chat-window .avatar-container img { border-radius: 50% !important; border: 2px solid var(--border) !important; }
.chat-window .message h1, .chat-window .message h2, .chat-window .message h3 { color: var(--accent) !important; margin: 12px 0 6px; font-weight: 600; }
.chat-window .message code { background: rgba(0,0,0,.3); padding: 2px 6px; border-radius: 4px; }
.chat-input { border-top: 1px solid var(--border) !important; background: var(--bg-secondary) !important; padding: 8px !important; }
.chat-input textarea {
    background: var(--bg-primary) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important; color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important; font-size: .9rem !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.chat-input textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-glow) !important; }
.chat-input button.primary, .chat-input button[class*="submit"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-teal)) !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; color: white !important;
    box-shadow: 0 2px 8px rgba(14,165,233,.3) !important;
}
.capability-tag { display: inline-block; background: var(--bg-tertiary); border: 1px solid var(--border);
    border-radius: 20px; padding: 4px 12px; font-size: .74rem; color: var(--text-muted); margin: 3px 2px; }
@media (max-width: 860px) { .main-row { flex-direction: column !important; } .main-row > .gr-column { min-width: 100% !important; } }
footer { opacity: .4 !important; }
"""

PRESET_QUERIES = [
    ("🦷 Early signs of gum disease?",  "What are the early signs of gum disease?"),
    ("🔍 Gingivitis vs Periodontitis",  "Explain the difference between gingivitis and periodontitis."),
    ("🏠 Treat sensitivity at home",    "How can I treat tooth sensitivity at home?"),
    ("🛡️ Preventing tooth decay",       "What causes tooth decay and how is it prevented?"),
]

def _add_user_message(message, history):
    user_text = (message.get("text") or "").strip()
    user_files = message.get("files") or []
    if not user_text and not user_files:
        return history, message, gr.update()
    for f in user_files:
        fp = f["path"] if isinstance(f, dict) else f
        history.append({"role": "user", "content": {"path": fp}})
    if user_text:
        history.append({"role": "user", "content": user_text})
    return history, message, gr.MultimodalTextbox(value=None)

def _generate_response(saved_msg, history):
    if saved_msg is None:
        yield history; return
    history.append({"role": "assistant", "content": "⏳ *Analyzing your query and searching the dental knowledge base…*"})
    yield history
    response = process_query(saved_msg, history)
    history[-1] = {"role": "assistant", "content": response}
    yield history

def _make_chip_handler(query_text):
    def handler(history):
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": "⏳ *Searching dental knowledge base…*"})
        yield history
        message = {"text": query_text, "files": []}
        response = process_query(message, history)
        history[-1] = {"role": "assistant", "content": response}
        yield history
    return handler

_THEME = gr.themes.Base(primary_hue=gr.themes.colors.sky, secondary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.slate)

with gr.Blocks(title="Dental Health AI Assistant") as demo:
    pending_msg = gr.State(None)

    gr.HTML("""<div class="app-header"><div class="header-content">
      <div class="header-left"><h1>🦷 Dental Health AI Assistant</h1>
      <p>Multimodal RAG · Powered by ChromaDB + Gemini 2.5 Flash</p></div>
      <div class="status-badges">
        <span class="status-badge"><span class="status-dot green"></span>Local RAG Active</span>
        <span class="status-badge"><span class="status-dot blue"></span>Gemini Vision Ready</span>
      </div></div></div>""")

    with gr.Row(elem_classes=["main-row"]):
        with gr.Column(scale=3, min_width=260):
            gr.HTML("""<div class="sidebar-panel"><h3>💡 How It Works</h3>
              <p>Upload a photo of your teeth or describe your symptoms, and the AI will:</p>
              <ul><li>Search a curated dental knowledge base</li>
              <li>Analyse uploaded images for visible concerns</li>
              <li>Provide detailed, evidence-backed explanations</li></ul>
              <p style="margin-top:10px;font-size:.78rem;color:var(--text-muted);">Supports <strong>JPG, PNG, WEBP, GIF</strong>.</p></div>""")

            chips = []
            for label, query in PRESET_QUERIES:
                btn = gr.Button(label, elem_classes=["chip-btn"], variant="secondary")
                chips.append((btn, query))

            gr.HTML("""<div class="sidebar-panel"><h3>📋 Knowledge Areas</h3>
              <div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:4px;">
              <span class="capability-tag">Dental Caries</span><span class="capability-tag">Gum Disease</span>
              <span class="capability-tag">Gingivitis</span><span class="capability-tag">Periodontitis</span>
              <span class="capability-tag">Oral Hygiene</span><span class="capability-tag">AI Dental Imaging</span>
              </div></div>""")

            gr.HTML("""<div class="sidebar-panel" style="background:var(--warning-bg);border-color:rgba(245,158,11,.3);">
              <div class="disclaimer-box" style="background:transparent;border:none;padding:0;">
              ⚠️ <strong>Educational use only.</strong> This assistant does <em>not</em> replace
              professional dental examination or treatment. Always consult a licensed dentist.</div></div>""")

        with gr.Column(scale=7, min_width=400, elem_classes=["chat-column"]):
            chatbot = gr.Chatbot(height=540, show_label=False, render_markdown=True,
                elem_classes=["chat-window"],
                placeholder="Ask a question or upload a dental image to get started…",
                avatar_images=(None, "https://api.dicebear.com/8.x/bottts/svg?seed=dental"),
                layout="bubble")
            msg = gr.MultimodalTextbox(placeholder="Describe your dental concern or upload an image…",
                show_label=False, file_types=["image"], elem_classes=["chat-input"],
                submit_btn=True, stop_btn=False)

    msg.submit(_add_user_message, [msg, chatbot], [chatbot, pending_msg, msg]).then(
        _generate_response, [pending_msg, chatbot], [chatbot])
    for btn, query_text in chips:
        btn.click(_make_chip_handler(query_text), [chatbot], [chatbot])

print("Launching on http://127.0.0.1:7861", flush=True)
demo.launch(server_name="127.0.0.1", server_port=7861, share=False,
            inbrowser=True, theme=_THEME, css=CUSTOM_CSS)
