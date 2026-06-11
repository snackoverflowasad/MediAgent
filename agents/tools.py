from langchain_core.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool
from rag.embeddings.embedding_model import get_embedding_model
from rag.vectorstore.chroma_store import get_chroma_db

def get_medical_retriever_tool() -> BaseTool:
    """Creates a LangChain retriever tool connected to the Chroma vector store."""
    embeddings = get_embedding_model()
    db = get_chroma_db(embedding=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    
    return create_retriever_tool(
        retriever=retriever,
        name="medical_knowledge_retriever",
        description=(
            "Use this tool to search and retrieve clinical context, medical guidelines, "
            "and healthcare documents. Input should be a specific medical or patient query."
        )
    )
