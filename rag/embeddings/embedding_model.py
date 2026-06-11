from langchain_openai import OpenAIEmbeddings
from rag.config import OPENAI_API_KEY

def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)