# src/helper.py (server-safe)
import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import (
    HuggingFaceInferenceAPIEmbeddings,
    HuggingFaceEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings 


def load_pdf_file(data: str):
    return DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader).load()

def text_split(extracted_data):
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20).split_documents(extracted_data)

def download_hugging_face_embeddings():
    # all-MiniLM-L6-v2 runs locally, no token needed
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
