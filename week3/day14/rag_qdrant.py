import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Load environment variables from .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fail early with a clear message if any key is missing
if not QDRANT_URL or not QDRANT_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing QDRANT_URL, QDRANT_API_KEY or GROQ_API_KEY in .env")

# ---------------------------------------------------------
# Connect to Qdrant
# ---------------------------------------------------------

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("connected to qdrant cloud")

# ---------------------------------------------------------
# Create Qdrant Collection
# ---------------------------------------------------------

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384  # all-MiniLM-L6-v2 outputs 384 numbers per text

# Delete collection if it already exists (fresh start every run)
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# Create collection
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE,
    ),
)

print(f"Created Collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# ---------------------------------------------------------
# Load our knowledge
# ---------------------------------------------------------

# Path(__file__).parent makes this work no matter where you run the script from
knowledge_path = Path(__file__).parent / "knowledge.txt"

# One document per non-empty line
with open(knowledge_path, "r", encoding="utf-8") as f:
    documents = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"Loaded {len(documents)} documents")

# ---------------------------------------------------------
# Create Embeddings
# ---------------------------------------------------------

print("Loading Embedding Model")

model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions

print("Embedding model ready")

# embeddings = all document vectors, vector = one row
embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} Embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# ---------------------------------------------------------
# Create Qdrant points
# ---------------------------------------------------------

points = []

for i, vector in enumerate(embeddings):
    point = PointStruct(
        id=i + 1,                       # unique id for each point
        vector=vector.tolist(),         # numpy array to plain list
        payload={"text": documents[i]}, # original text stored with the vector
    )
    points.append(point)

# ---------------------------------------------------------
# Upload to Qdrant
# ---------------------------------------------------------

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)

print(f"Uploaded {len(points)} documents to Qdrant")

# ---------------------------------------------------------
# Search Qdrant
# ---------------------------------------------------------

def search(query, top_k=3):
    # Convert the question into an embedding (same model as the documents)
    query_vector = model.encode(query).tolist()

    # Search Qdrant for the most similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    return results

# ---------------------------------------------------------
# Test Search (optional, delete once it works)
# ---------------------------------------------------------

test_query = "how many vacations do i get?"

test_results = search(test_query, top_k=3)

print("\nSearch Results:")

for result in test_results:
    print(f"Score: {result.score:.3f}")
    print(result.payload["text"])
    print()

# ---------------------------------------------------------
# Connect to Groq
# ---------------------------------------------------------

groq_client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# Ask the LLM
# ---------------------------------------------------------

def ask_llm(question, context):
    # Prompt forces the model to answer only from the retrieved context
    prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
"I dont know based on the provided information."
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",  # must be a string, so it needs quotes
                "content": prompt,
            }
        ],
    )

    return response.choices[0].message.content

# ---------------------------------------------------------
# Complete RAG Pipeline
# ---------------------------------------------------------

question = "How many vacations do i get?"

# 1. retrieve the top matching chunks
results = search(question, top_k=3)

# 2. extract the text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)

# 3. send question + context to the llm
answer = ask_llm(question, context)

print("\nFinal Answer")
print(answer)