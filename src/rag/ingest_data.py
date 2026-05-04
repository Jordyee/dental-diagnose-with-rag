import os
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ensure_directory_exists(directory_path):
    """Ensure that a directory exists; if not, create it."""
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        except Exception as e:
            print(f"Error creating directory {directory_path}: {e}")

def get_base_dir():
    """Return the base directory of the project (assuming this script is in src/rag/)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))

def load_documents(data_path):
    print(f"Attempting to load documents from: {data_path}")
    documents = []
    
    # We will use explicit loaders for PDF and TXT
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=False)
    txt_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader, show_progress=False)
    
    try:
        print("Loading PDF files...")
        pdf_docs = pdf_loader.load()
        print(f"Successfully loaded {len(pdf_docs)} PDF pages/documents.")
        documents.extend(pdf_docs)
    except Exception as e:
        print(f"Error loading PDF documents (could be missing/corrupt files): {e}")
        
    try:
        print("Loading TXT files...")
        txt_docs = txt_loader.load()
        print(f"Successfully loaded {len(txt_docs)} TXT pages/documents.")
        documents.extend(txt_docs)
    except Exception as e:
        print(f"Error loading TXT documents (could be missing/corrupt files): {e}")

    return documents

def main():
    print("--- Starting RAG Ingestion Pipeline ---")
    
    # 1. Environment
    print("Loading environment variables...")
    load_dotenv()
    
    if not os.environ.get("GOOGLE_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable not found! ")
        print("Please ensure it's set in your .env file or environment variables.")

    # Set up paths
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "data", "raw_pdfs")
    persist_dir = os.path.join(base_dir, "data", "chroma_db_gigi")
    
    # Check and create directories if necessary
    print("Checking directories...")
    ensure_directory_exists(data_dir)
    ensure_directory_exists(persist_dir)
    
    # 2. Loading
    documents = load_documents(data_dir)
    
    if not documents:
        print("No documents were loaded. Pipeline finished early.")
        print(f"Please add .pdf or .txt files to {data_dir} and try again.")
        return

    # 3. Chunking
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    try:
        chunks = text_splitter.split_documents(documents)
        # Filter out any chunks with empty content to prevent embedding errors
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        print(f"Successfully split documents into {len(chunks)} valid chunks.")
    except Exception as e:
        print(f"Error during document chunking: {e}")
        return

    # 4 & 5. Embedding & Vector Store
    print("Initializing HuggingFace Embeddings...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        print(f"Error initializing embeddings model: {e}")
        return

    print(f"Creating and persisting ChromaDB at: {persist_dir}")
    try:
        # Saving in batches to prevent API rate limits or payload size limits
        batch_size = 100
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vector_store.add_documents(batch)
            
        print("Data successfully embedded and saved to vector store.")
    except Exception as e:
        print(f"Error creating or persisting vector store: {e}")
        return

    print("--- Pipeline Execution Complete ---")

if __name__ == "__main__":
    main()
