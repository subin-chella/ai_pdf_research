import os
# Disable ChromaDB telemetry to avoid telemetry errors
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_and_split_pdf(file_path):
    print(f"Loading and splitting PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from the PDF.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def store_chunks_in_chroma(chunks, persist_directory):
    print(f"Storing {len(chunks)} chunks in ChromaDB at {persist_directory}.")
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        # Note: db.persist() is no longer needed in Chroma 0.4.x - docs are automatically persisted
        print("Chunks stored successfully.")
        return db
    except Exception as e:
        print(f"Warning: Could not store chunks in ChromaDB: {e}")
        print("Continuing without vector storage for now.")
        return None
    
def get_retriever(persist_directory):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        retriever = db.as_retriever()
        print("Retriever created successfully.")
        return retriever
    except Exception as e:
        print(f"Error loading ChromaDB: {e}")
        return None    
