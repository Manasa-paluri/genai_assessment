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

# Set page configuration
st.set_page_config(page_title="PDF Q&A with Gemini", layout="wide")

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'text_chunks' not in st.session_state:
    st.session_state.text_chunks = []
if 'pdf_uploaded' not in st.session_state:
    st.session_state.pdf_uploaded = False
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None


if not GOOGLE_API_KEY:
    st.warning("Please set the GOOGLE_API_KEY in Streamlit secrets or environment variables!")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# Initialize sentence transformer model
@st.cache_resource
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
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

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) > 100:  # Only add chunks that have substantial content
            chunks.append(chunk)
    return chunks

def create_faiss_index(text_chunks, embedder):
    """Create FAISS index from text chunks"""
    # Generate embeddings for all text chunks
    embeddings = embedder.encode(text_chunks)

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Create FAISS index - using IndexFlatIP for inner product (cosine similarity)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index

def query_document(question, index, text_chunks, embedder, top_k=3):
    """Query FAISS index to find relevant document chunks"""
    # Generate embedding for question
    question_embedding = embedder.encode([question])

    # Normalize embedding
    faiss.normalize_L2(question_embedding)

    # Search the index
    scores, indices = index.search(question_embedding, top_k)

    # Get the relevant text chunks
    relevant_chunks = [text_chunks[idx] for idx in indices[0]]

    # Join the chunks
    context = "\n\n".join(relevant_chunks)
    return context

def get_gemini_response(question, context):
    """Get response from Gemini model based on context"""
    prompt = f"""
    You are a helpful assistant. Your task is to answer questions based on the provided document context.
    You should provide a concise and accurate answer based on the information in the context.
    Based on the following document context, please answer the questions like what is in the document,explain pdf in short and any other questions ,if possible give answer in language they want. 
    If the document doesn't contain information to answer the question, respond with: "I couldn't find relevant information in the document." and provide summarisation of document in detail

    Context:
    {context}

    Question: {question}
    """

    response = model.generate_content(prompt)
    return response.text

# UI Components
st.title("PDF Question Answering System")

# Load the embedder
if st.session_state.embedder is None:
    with st.spinner("Loading embedding model..."):
        st.session_state.embedder = load_sentence_transformer()

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None and not st.session_state.pdf_uploaded:
        with st.spinner("Processing PDF..."):
            # Extract text from PDF
            text_content = extract_text_from_pdf(uploaded_file)

            # Split text into chunks
            chunks = split_text_into_chunks(text_content)
            st.info(f"Document processed: {len(chunks)} text chunks extracted")

            # Create FAISS index
            st.session_state.text_chunks = chunks
            st.session_state.faiss_index = create_faiss_index(chunks, st.session_state.embedder)
            st.session_state.pdf_uploaded = True
            st.session_state.pdf_name = uploaded_file.name
            st.success(f"Successfully uploaded and indexed: {uploaded_file.name}")

    if st.session_state.pdf_uploaded:
        st.success(f"Current document: {st.session_state.pdf_name}")
        if st.button("Clear Document"):
            st.session_state.chat_history = []
            st.session_state.faiss_index = None
            st.session_state.text_chunks = []
            st.session_state.pdf_uploaded = False
            st.session_state.pdf_name = None
            st.experimental_rerun()

# Main chat interface
if st.session_state.pdf_uploaded:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about the document")
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})

        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching document and generating response..."):
                # Retrieve relevant context using FAISS
                context = query_document(
                    user_question, 
                    st.session_state.faiss_index, 
                    st.session_state.text_chunks, 
                    st.session_state.embedder
                )

                # Get response from Gemini
                response = get_gemini_response(user_question, context)

                # Display response
                st.write(response)

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
    st.info("Please upload a PDF document to start asking questions.")

# Footer
st.markdown("---")
st.caption("PDF Q&A System using FAISS and Google Gemini")








