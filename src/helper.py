# src/helper.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings  # <-- keep this

def load_pdf_file(data):
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def text_split(extracted_data):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(extracted_data)

def download_hugging_face_embeddings():
    # local model, no token needed
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
