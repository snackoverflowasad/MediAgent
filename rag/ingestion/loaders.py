from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from typing import List

def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()

def load_txt(file_path: str) -> List[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()
