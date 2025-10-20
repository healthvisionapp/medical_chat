# src/helper.py

import os

# --- Loaders & splitters (keep your originals) ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Embeddings ---
# We'll prefer the Inference API (tiny, no torch) if a HF token is present.
# Fallback to local HuggingFaceEmbeddings (requires torch) for local dev.
try:
    # Newer, community path
    from langchain_community.embeddings import (
        HuggingFaceInferenceEmbeddings,
        HuggingFaceEmbeddings,
    )
except Exception:
    # Older path (kept for compatibility just in case)
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore
    HuggingFaceInferenceEmbeddings = None  # type: ignore


# ---------------------------
# PDF loading (no change)
# ---------------------------
def load_pdf_file(data: str):
    """
    Load all PDFs from a directory path `data` into LangChain Documents.
    Example: data='Data/'  (trailing slash OK)
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# ---------------------------
# Text splitting (no change)
# ---------------------------
def text_split(extracted_data):
    """
    Split documents into small chunks for embedding/retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# ---------------------------
# Embeddings (token-aware)
# ---------------------------
def download_hugging_face_embeddings():
    """
    Return embeddings compatible with your Pinecone index (384-dim)
    using 'sentence-transformers/all-MiniLM-L6-v2'.

    - On Koyeb (production): if HUGGINGFACEHUB_API_TOKEN is present,
      use HuggingFace Inference API (no PyTorch, small image).
    - On local dev: fall back to on-device HuggingFaceEmbeddings
      (requires torch).
    """
    model_id = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim (matches your index)
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()

    # Prefer Inference API if token provided and class available
    if hf_token and 'HuggingFaceInferenceEmbeddings' in globals() and HuggingFaceInferenceEmbeddings:
        return HuggingFaceInferenceEmbeddings(
            api_key=hf_token,
            model_name=model_id
        )

    # Fallback: local embeddings (will pull torch; fine on your laptop)
    return HuggingFaceEmbeddings(model_name=model_id)
