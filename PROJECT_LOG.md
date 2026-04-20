# Project Log

[2026-04-21] | RAG Ingestion Pipeline | Created `src/rag/ingest_data.py` to ingest, chunk, embed, and store PDF and TXT data into ChromaDB | Implementing multimodal RAG data ingestion pipeline according to INSTRUCTIONS_FOR_AGENT.md specifications.
[2026-04-21] | RAG Ingestion Pipeline Fixes | Updated `src/rag/ingest_data.py` | Added `autodetect_encoding=True` to TextLoader and improved vector store persistence by processing in batches and filtering empty chunks to fix embedding runtime errors.
[2026-04-21] | RAG Architecture Update | Updated `src/rag/ingest_data.py` | Migrated to `HuggingFaceEmbeddings` (sentence-transformers/all-MiniLM-L6-v2) to resolve Google API 404 error, refactored Chroma import to use `langchain_chroma`, and reverted TextLoader kwargs.
