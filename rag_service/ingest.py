import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

DB_DIR = "./chroma_db"
DATA_DIR = "./data"

def get_loader(file_path):
    """Factory function to route files to the correct parser based on extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    else:
        print(f"Skipping unsupported file type: {file_path}")
        return None

def ingest_documents():
    print("Starting ingestion process...")
    # Find all files in the data directory
    # os.path.join ensures cross-platform compatibility (Linux/Windows)
    all_files = glob.glob(os.path.join(DATA_DIR, "*.*"))
    
    docs = []
    for file_path in all_files:
        loader = get_loader(file_path)
        if loader:
            print(f"Loading: {file_path}")
            docs.extend(loader.load())

    if not docs:
        print("No readable documents found in the data directory.")
        return

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Embed and persist to disk
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Embedding and saving to ChromaDB...")
    
    # By providing a persist_directory, Chroma saves the DB to your hard drive
    Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    print("Ingestion complete. Database is ready.")

if __name__ == "__main__":
    ingest_documents()