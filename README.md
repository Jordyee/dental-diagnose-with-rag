# 🦷 Dental Health AI Assistant

A **Multimodal RAG (Retrieval-Augmented Generation)** application for dental health education and early diagnosis assistance. Built with LangChain, ChromaDB, HuggingFace Embeddings, and Google Gemini 2.5 Flash.

> **Disclaimer:** This application is for **educational purposes only** and does not replace professional dental examination or treatment. Always consult a licensed dentist.

---

## Features

- **Text + Image input** — describe symptoms or upload a dental photo
- **RAG pipeline** — retrieves relevant context from 30+ dental journals and books before answering
- **Gemini 2.5 Flash** — multimodal LLM that can analyze dental images
- **Dark-themed Gradio UI** — responsive two-column layout with quick-start chips
- **Export chat** — download conversation history as a `.txt` file

---

## Tech Stack

| Component | Technology |
|---|---|
| UI Framework | [Gradio](https://gradio.app) |
| LLM | Google Gemini 2.5 Flash (via `langchain-google-genai`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Database | [ChromaDB](https://www.trychroma.com) (local, persisted) |
| RAG Orchestration | [LangChain](https://python.langchain.com) |
| Environment | Python 3.10+ |

---

## Project Structure

```
dental-diagnose-with-rag/
├── data/
│   ├── raw_pdfs/          # Source documents (PDFs & TXTs) for the knowledge base
│   └── chroma_db_gigi/    # Generated vector DB — created by ingest_data.py (gitignored)
├── src/
│   ├── rag/
│   │   └── ingest_data.py # Pipeline: load → chunk → embed → store in ChromaDB
│   └── ui/
│       └── app.py         # Main Gradio application
├── .env.example           # Template for environment variables
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Prerequisites

- **Python 3.10 or newer** ([download](https://www.python.org/downloads/))
- **Git** ([download](https://git-scm.com/downloads))
- **Google API Key** with access to the Gemini API ([get one here](https://aistudio.google.com/app/apikey))

---

## Setup & Installation

Follow these steps **exactly** to run the project on your own machine.

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/dental-diagnose-with-rag.git
cd dental-diagnose-with-rag
```

### Step 2 — Create and activate a virtual environment

**Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> This may take a few minutes on first run as it downloads the HuggingFace embedding model and all LangChain packages.

### Step 4 — Set up your API key

Copy the example environment file and fill in your Google API key:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` in any text editor and replace `your_google_api_key_here` with your real key:

```
GOOGLE_API_KEY=AIzaSy...your_actual_key...
```

> **How to get a Google API Key:**
> 1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
> 2. Sign in with your Google account
> 3. Click **"Create API key"**
> 4. Copy the generated key and paste it into `.env`

### Step 5 — Build the vector database

This step processes all PDF and TXT files in `data/raw_pdfs/` and stores their embeddings in a local ChromaDB database. **You only need to run this once.**

```bash
python src/rag/ingest_data.py
```

Expected output:
```
--- Starting RAG Ingestion Pipeline ---
Loading environment variables...
Loading PDF files...
Successfully loaded X PDF pages/documents.
...
Data successfully embedded and saved to vector store.
--- Pipeline Execution Complete ---
```

> The process can take 5–15 minutes depending on your machine. A `data/chroma_db_gigi/` folder will be created.

### Step 6 — Run the application

```bash
python src/ui/app.py
```

The app will open automatically in your browser at [http://127.0.0.1:7860](http://127.0.0.1:7860).

---

## How to Use

1. **Type a question** in the chat box (e.g., *"What are the symptoms of gingivitis?"*)
2. **Upload an image** (optional) — attach a dental photo for visual analysis
3. **Use Quick Start chips** on the left sidebar to try preset questions
4. **Export** your conversation using the "Export as TXT" button

---

## Troubleshooting

### `GOOGLE_API_KEY not found` error
Make sure your `.env` file exists in the project root and contains the correct key. The `.env` file should NOT have quotes around the key:
```
# Correct:
GOOGLE_API_KEY=AIzaSyXXXXXXXXX

# Wrong:
GOOGLE_API_KEY="AIzaSyXXXXXXXXX"
```

### `ModuleNotFoundError` after `pip install`
Ensure your virtual environment is activated (you should see `(venv)` in your terminal). Re-run `pip install -r requirements.txt` with the venv active.

### ChromaDB error on startup
The `data/chroma_db_gigi/` folder is missing or empty. Run the ingestion step again:
```bash
python src/rag/ingest_data.py
```

### Port 7860 already in use
Another Gradio app is running. Either stop it or change the port in `src/ui/app.py`:
```python
app.launch(server_port=7861, ...)  # change to any free port
```

---

## Re-building the Knowledge Base

If you add new documents to `data/raw_pdfs/`, delete the old ChromaDB folder and re-run ingestion:

```bash
# Windows
rmdir /s /q data\chroma_db_gigi

# macOS / Linux
rm -rf data/chroma_db_gigi

# Then re-ingest
python src/rag/ingest_data.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push and open a Pull Request

---

## License

This project is developed for academic purposes (Semester 4 — Expert System Course).
