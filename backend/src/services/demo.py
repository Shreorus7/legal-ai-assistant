import logging
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ------------------------------------------
# Setup logger
# ------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

log.info("🚀 Starting ChromaDB + SentenceTransformer embedding process...")

# ------------------------------------------
# Initialize ChromaDB
# ------------------------------------------
log.info("🔧 Initializing ChromaDB client...")
client = PersistentClient(path="./chroma_db")

# ------------------------------------------
# Load sentence-transformer model
# ------------------------------------------
log.info("📥 Loading embedding model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------
# Define documents
# ------------------------------------------
docs = [
    "This is a document about pineapple",
    "This is a document about oranges"
]
ids = ["id1", "id2"]

log.info(f"📝 Loaded {len(docs)} documents")

# ------------------------------------------
# Generate embeddings
# ------------------------------------------
log.info("🧠 Generating embeddings...")
embeddings = model.encode(docs, convert_to_numpy=True).tolist()
log.info("✅ Embeddings generated.")

# ------------------------------------------
# Store into ChromaDB
# ------------------------------------------
collection_name = "my_collection"
log.info(f"📦 Getting or creating collection: '{collection_name}'")
collection = client.get_or_create_collection(collection_name)

log.info("📤 Adding documents to the collection...")
collection.add(ids=ids, documents=docs, embeddings=embeddings)
log.info("✅ Documents added successfully.")

# ------------------------------------------
# Query to test retrieval
# ------------------------------------------
query = "Tell me something about tropical fruit"
log.info(f"🔍 Performing test query: '{query}'")
results = collection.query(query_texts=[query], n_results=2)

log.info("📊 Query Results:")
for idx, doc in enumerate(results.get("documents", [[]])[0]):
    log.info(f"Result {idx + 1}: {doc}")

log.info("🏁 Done.")
