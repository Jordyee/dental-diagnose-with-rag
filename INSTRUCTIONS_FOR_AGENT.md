# ROLE DEFINITION
You are a Senior AI Engineer specializing in LangChain, RAG (Retrieval-Augmented Generation), and Vector Databases.

# PROJECT CONTEXT
- Project Name: dental-diagnose-with-rag
- Goal: Build an educational and early diagnosis system for dental health (specifically dental calculus and caries) using Multimodal RAG.
- Data Location: `data/raw_pdfs/` (contains PDF and TXT files).
- Vector DB: ChromaDB (Local).
- Embedding Model: GoogleGenerativeAIEmbeddings (model: 'models/embedding-001').

# TASK: RAG INGESTION PIPELINE
Create a Python script at `src/rag/ingest_data.py` with the following workflow:
1. Environment: Load API keys from `.env` using `python-dotenv`.
2. Loading: Use LangChain's `DirectoryLoader` to read both `.pdf` and `.txt` files recursively from `data/raw_pdfs/`.
3. Chunking: Use `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`.
4. Embedding & Vector Store: Initialize and persist a ChromaDB instance at `data/chroma_db_gigi/`.
5. Persistence: Ensure data is persistently saved to disk.

# RULES FOR DOCUMENTATION (CRITICAL)
Every time you complete a task or modify code:
- You MUST update the `PROJECT_LOG.md` file.
- Log Format: [Date] | [Component] | [Description of Changes] | [Reason/Context].
- Ensure this documentation is clear so future AI agents can resume work without losing context.

# CONSTRAINTS (DOS & DON'TS)
- DO: Implement error handling (try-except) for corrupted files during loading.
- DO: Add console logging (print statements) so the user knows the execution progress.
- DO: Use `os.path` for folder paths to ensure cross-platform compatibility (Windows/Linux/Mac).
- DON'T: Do not use any embedding model other than Google Generative AI unless explicitly instructed.
- DON'T: Do not push or commit to GitHub; only create/modify local files.
- DON'T: Do not assume directories exist; write code to check and create them if necessary (`os.makedirs`).

# OUTPUT EXPECTATION
A fully functional `src/rag/ingest_data.py` ready to be executed, and an updated `PROJECT_LOG.md`.