import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
import PyPDF2
import time

# Function to load PDF and extract text with progress bar
def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    total_pages = len(reader.pages)
    text = ""
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, page in enumerate(reader.pages):
        text += page.extract_text()
        progress_bar.progress((i + 1) / total_pages)
        status_text.text(f"Reading page {i + 1} of {total_pages}")
        time.sleep(0.1)  # Simulate delay for UI update (can be removed in production)

    progress_bar.empty()  # Remove the progress bar when done
    status_text.text("PDF loading complete.")
    return text

# Split the extracted text into chunks
def split_pdf_text(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_text(pdf_text)
    return [Document(page_content=chunk) for chunk in split_texts]

# Create the RAG setup
def create_retriever(documents):
    embeddings = OllamaEmbeddings(model="gemma2:2b")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

# Function to call the Ollama gemma2:2b model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='gemma2:2b', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain function
def rag_chain(question, retriever):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)

# Streamlit app
st.title("RAG with gemma2:2b")
st.write("Ask questions about the provided context")

# Upload the PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Display a loading spinner while processing
    with st.spinner("Processing PDF..."):
        # Load and split PDF with progress bar
        pdf_text = load_pdf(uploaded_file)
        documents = split_pdf_text(pdf_text)

        # Create retriever
        retriever = create_retriever(documents)

    # Input for the question
    question = st.text_input("Enter your question here...")

    if question:
        with st.spinner("Retrieving answer..."):
            # Get the answer using RAG
            answer = rag_chain(question, retriever)
            st.write("Answer:", answer)