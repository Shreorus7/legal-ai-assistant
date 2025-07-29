import os
import shutil

# Optional: wipe and retry from a clean DB
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

from chromadb import EphemeralClient
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger()

log.info("🚀 Starting with Ephemeral")

client = EphemeralClient()
collection = client.get_or_create_collection("test_col")

docs = ["apple is red", "banana is yellow"]
ids = ["1", "2"]
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs, convert_to_numpy=True).tolist()

log.info("📤 Adding")
collection.add(ids=ids, documents=docs, embeddings=embeddings)
log.info("✅ Success")