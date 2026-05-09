import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="RAG PDF Chat", layout="wide")

st.title("📄 RAG PDF Pipeline App")

# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    # Save uploaded file
    pdf_path = f"temp_{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully")

    # Load PDF
    # -----------------------------
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    st.write(f"Loaded {len(documents)} pages")

    # Split Documents
    # -----------------------------
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    st.write(f"Created {len(chunks)} chunks")


    # Embedding Model
    # -----------------------------
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # -----------------------------
    # ChromaDB Setup
    # -----------------------------
    client = chromadb.PersistentClient(path="vector_store")

    collection = client.get_or_create_collection(
        name="pdf_collection"
    )

     # Store Embeddings
    # -----------------------------
    for chunk in chunks:
        embedding = model.encode(chunk.page_content).tolist()

        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[chunk.page_content],
            embeddings=[embedding]
        )

    st.success("Embeddings stored successfully")

# Search Section
    # -----------------------------
    query = st.text_input("Ask Question")

    if query:

        query_embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        st.subheader("Top Results")

        for doc in results["documents"][0]:
            st.write(doc)
            st.divider()
