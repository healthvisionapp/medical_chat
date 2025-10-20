# src/helper.py
import os

# New, non-deprecated imports
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Correct class name for HF Inference API:
from langchain_community.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceEmbeddings,
)

def load_pdf_file(data: str):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)

def download_hugging_face_embeddings():
    model_id = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim (matches your Pinecone index)
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
    if hf_token:
        # Use HF Inference API on Koyeb (no torch needed)
        return HuggingFaceInferenceAPIEmbeddings(api_key=hf_token, model_name=model_id)
    # Local dev fallback (requires sentence_transformers on your laptop only)
    return HuggingFaceEmbeddings(model_name=model_id)
