from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from typing import List
from rag.config import CHROMA_PERSIST_DIR

def store_chroma(chunks: List[Document], embedding: Embeddings, persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Stores a list of document chunks into Chroma DB vector store using Chroma.from_documents."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )

def get_chroma_db(embedding: Embeddings, persist_dir: str = CHROMA_PERSIST_DIR) -> Chroma:
    """Loads and returns the existing Chroma DB instance."""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )