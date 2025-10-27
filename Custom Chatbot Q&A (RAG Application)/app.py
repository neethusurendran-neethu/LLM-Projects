import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(page_title="Multi-File RAG Chatbot", page_icon="📚", layout="wide")

st.title("📚 Multi-File RAG Chatbot")
st.write("Upload multiple documents and ask questions powered by LLMs!")

# Add instructions for users
with st.expander("ℹ️ How to use this application"):
    st.markdown("""
    1. Upload one or more documents (PDF or TXT) using the sidebar
    2. Click "Process Files" to create the vector store
    3. Ask questions about your documents in the chat area below
    """)

# ---------------------------------------
# Session State Init
# ---------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ---------------------------------------
# Load Documents
# ---------------------------------------
def load_documents(uploaded_files):
    documents = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        try:
            if file.type == "text/plain":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif file.type == "application/pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                continue

            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            documents.extend(docs)
        except Exception as e:
            st.error(f"❌ Error loading {file.name}: {e}")
        finally:
            os.unlink(tmp_path)

    return documents

# ---------------------------------------
# Create Vectorstore
# ---------------------------------------
def create_vectorstore(documents):
    if not documents:
        return None

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None

# ---------------------------------------
# Create QA Chain
# ---------------------------------------
def create_qa_chain(vectorstore):
    try:
        # Using a local model from HuggingFace Hub as an alternative to ChatOpenAI
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.1, "max_length": 512}
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff",
            return_source_documents=True,
        )
    except Exception as e:
        st.error(f"❌ Error creating QA chain: {str(e)}")
        return None

# ---------------------------------------
# Sidebar: Upload & Process Files
# ---------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_files = st.file_uploader(
        "Upload Files", type=["txt", "pdf"], accept_multiple_files=True
    )

    if st.button("🚀 Process Files"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                documents = load_documents(uploaded_files)
                st.session_state.vectorstore = create_vectorstore(documents)

                if st.session_state.vectorstore:
                    st.session_state.qa_chain = create_qa_chain(st.session_state.vectorstore)
                    st.success("✅ Files processed!")
                else:
                    st.error("❌ No content to process")
        else:
            st.warning("⚠️ Please upload at least one file")

    if st.button("🧹 Clear All"):
        st.session_state.vectorstore = None
        st.session_state.qa_chain = None
        st.success("Session cleared!")

# ---------------------------------------
# Main Chat Area
# ---------------------------------------
if st.session_state.qa_chain:
    st.subheader("💬 Ask Questions About Your Documents")

    query = st.text_input("Type your question here:")

    if query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain(query)

        st.markdown("### ✅ Answer:")
        st.write(result["result"])

        if result.get("source_documents"):
            st.markdown("### 📄 Sources:")
            for doc in result["source_documents"]:
                st.caption(f"- {doc.metadata.get('source', 'Unknown')}")
else:
    st.info("👆 Upload and process documents from the sidebar to start chatting.")
