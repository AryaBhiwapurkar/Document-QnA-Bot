import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from llm import get_embeddings


DB_PATH = "db"


def ingest_pdf(file_path, embeddings=None):

    if embeddings is None:
        embeddings = get_embeddings()

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return False

    os.makedirs(DB_PATH, exist_ok=True)

    print("Loading PDF...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    if not docs:
        print("Error: No content extracted from PDF.")
        return False

    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"Total Chunks: {len(chunks)}")

    print("Creating embeddings...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)

    print("FAISS DB Saved ✅")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest_pdf(sys.argv[1])