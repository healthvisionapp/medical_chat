# store_index.py

from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

# --- 1) Env & client ---
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set")

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"
dimension = 384  # all-MiniLM-L6-v2

# --- 2) Make sure the index exists (create if missing) ---
existing = [i["name"] for i in pc.list_indexes().indexes]
if index_name not in existing:
    print(f"[pinecone] creating index '{index_name}' (dim={dimension})...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"[pinecone] index '{index_name}' already exists, will reuse it.")

# optional: wait until ready (usually quick)
for _ in range(30):
    try:
        d = pc.describe_index(index_name)
        if d and d.status and d.status.get("ready"):
            break
    except Exception:
        pass
    time.sleep(1)
print("[pinecone] index is ready.")

# --- 3) Load your PDF(s), split, embeddings ---
# NOTE: your PDF is inside the 'Data/' folder
docs = load_pdf_file(data="Data/")
print(f"[data] loaded {len(docs)} document(s)")

chunks = text_split(docs)
print(f"[data] split into {len(chunks)} chunks")

embeddings = download_hugging_face_embeddings()
print("[embeddings] ready")

# --- 4) Upsert chunks into Pinecone ---
print("[upsert] writing vectors to Pinecone...")
_ = PineconeVectorStore.from_documents(
    documents=chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("[done] upsert complete. Check vector count in the Pinecone dashboard.")
