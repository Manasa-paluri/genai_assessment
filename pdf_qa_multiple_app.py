
import streamlit as st
import os
import tempfile
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import numpy as np
import faiss
import uuid
import pickle
from typing import List, Dict, Any
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set page configuration for full screen width
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")

# Initialize session state variables if they don't exist
if 'pdfs' not in st.session_state:
    st.session_state.pdfs = {}  # Store multiple PDFs with their chunks and indices

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name

    text_content = ""
    with open(temp_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text_content += page.extract_text() + "\n\n"

    os.unlink(temp_file_path)
    return text_content

# Split text into chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

# Create FAISS index
def create_faiss_index(text_chunks, embedder):
    embeddings = embedder.encode(text_chunks)
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

# Query across multiple PDFs
def query_documents(question, top_k=3):
    all_relevant_chunks = []
    for pdf_name, pdf_data in st.session_state.pdfs.items():
        index = pdf_data["index"]
        chunks = pdf_data["chunks"]

        # Generate embedding for the question
        question_embedding = st.session_state.embedder.encode([question])
        faiss.normalize_L2(question_embedding)

        # Search the index for relevant chunks
        scores, indices = index.search(question_embedding, top_k)
        relevant_chunks = [chunks[idx] for idx in indices[0]]
        all_relevant_chunks.extend(relevant_chunks)

    # Combine all relevant chunks from all documents
    context = "\n\n".join(all_relevant_chunks)
    return context

# Main App
st.title("PDF Question Answering System (Multiple PDFs)")

if 'embedder' not in st.session_state:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = load_sentence_transformer()

uploaded_files = st.file_uploader("Choose PDF files", type=['pdf'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.pdfs:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                text_content = extract_text_from_pdf(uploaded_file)
                chunks = split_text_into_chunks(text_content)
                index = create_faiss_index(chunks, st.session_state.embedder)
                st.session_state.pdfs[uploaded_file.name] = {
                    "chunks": chunks,
                    "index": index
                }
                st.success(f"Successfully processed and indexed: {uploaded_file.name}")
        else:
            st.info(f"{uploaded_file.name} is already processed.")

# Query handling for multiple PDFs
for pdf_name in st.session_state.pdfs.keys():
    st.markdown(f"**{pdf_name}**")
    user_question = st.text_input(f"Ask a question about {pdf_name}")
    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        with st.spinner(f"Searching {pdf_name}..."):
            context = query_documents(user_question)
            response = get_gemini_response(user_question, context)

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.write(response)

