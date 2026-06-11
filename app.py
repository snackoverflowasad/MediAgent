import streamlit as st
import tempfile
import os
from pathlib import Path
from rag.ingestion.loaders import load_pdf, load_txt
from rag.ingestion.splitters import split_documents
from rag.embeddings.embedding_model import get_embedding_model
from rag.vectorstore.chroma_store import store_chroma
from agents.medical_agent import create_medical_agent

st.set_page_config(
    page_title="MediAgent - RAG-Powered Healthcare Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #161b22;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #00f2fe, #4facfe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
</style>
""", unsafe_allow_html=True)

st.title("MediAgent: RAG-Powered Healthcare Assistant")
st.markdown("---")

if "agent_executor" not in st.session_state:
    try:
        st.session_state.agent_executor = create_medical_agent()
    except Exception as e:
        st.error(f"Failed to initialize Agent: {e}")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Ingestion Hub")
    st.markdown("Upload medical guidelines, clinical trials, or patient records to update the knowledge base.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])
    
    if uploaded_file:
        file_ext = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            
        try:
            with st.spinner("Processing document..."):
                if file_ext == ".pdf":
                    docs = load_pdf(temp_file_path)
                else:
                    docs = load_txt(temp_file_path)
                
                chunks = split_documents(docs)
                embeddings = get_embedding_model()
                store_chroma(chunks, embeddings)
                
                st.success(f"Successfully ingested {len(chunks)} chunks!")
 
                st.session_state.agent_executor = create_medical_agent()
                
        except Exception as e:
            st.error(f"Ingestion error: {e}")
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

st.subheader("Clinical Chat Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a clinical question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if "agent_executor" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing queries & retrieving context..."):
                try:
                    response = st.session_state.agent_executor.invoke({
                        "input": user_query,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    output_text = response["output"]
                    st.markdown(output_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                    
                    st.session_state.chat_history.append(("human", user_query))
                    st.session_state.chat_history.append(("ai", output_text))
                    
                except Exception as e:
                    st.error(f"Error executing agent query: {e}")
    else:
        st.error("Agent executor is not initialized. Please check your configurations/OpenAI keys.")
