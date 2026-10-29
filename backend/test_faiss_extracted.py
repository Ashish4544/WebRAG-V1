from dotenv import load_dotenv

load_dotenv()

import requests
import trafilatura
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# ==========================================
# 1. Load webpage
# ==========================================

url = "https://m.cricbuzz.com/live-cricket-scores/151532/ind-vs-wi-1st-odi-west-indies-tour-of-india-2026"

response = requests.get(
    url,
    headers={"User-Agent": "WebRAG/1.0"},
    timeout=15
)

response.raise_for_status()

html = response.text

print("HTML characters:", len(html))


# ==========================================
# 2. Extract meaningful webpage content
# ==========================================

text = trafilatura.extract(
    html,
    include_links=False,
    include_images=False,
    include_tables=True
)

if not text:
    raise ValueError("Could not extract webpage content")

print("Extracted characters:", len(text))


# ==========================================
# 3. Split content into chunks
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.create_documents([text])

print("Total chunks:", len(chunks))


# ==========================================
# 4. Create embeddings
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)


# ==========================================
# 5. Create FAISS index
# ==========================================

dimension = embeddings.shape[1]

# Inner Product on normalized vectors
# is equivalent to cosine similarity

index = faiss.IndexFlatIP(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)


# ==========================================
# 6. Create BM25 index
# ==========================================

tokenized_chunks = [
    chunk.lower().split()
    for chunk in texts
]

bm25 = BM25Okapi(tokenized_chunks)


# ==========================================
# 7. User query
# ==========================================

query = "How many runs did they have after 20 overs?"

print("\nQuery:", query)


# ==========================================
# 8. Create query embedding
# ==========================================

query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

query_embedding = np.array(
    query_embedding
).astype("float32")


# ==========================================
# 9. FAISS semantic search
# ==========================================

k = min(5, len(chunks))

faiss_scores, faiss_indices = index.search(
    query_embedding,
    k
)

print("\n" + "=" * 60)
print("FAISS RESULTS")
print("=" * 60)

for rank, (score, idx) in enumerate(
    zip(faiss_scores[0], faiss_indices[0]),
    start=1
):

    print(f"\n--- FAISS Rank {rank} ---")
    print("Chunk index:", idx)
    print("Cosine similarity:", score)
    print(chunks[idx].page_content)


# ==========================================
# 10. BM25 keyword search
# ==========================================

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(tokenized_query)

bm25_indices = np.argsort(
    bm25_scores
)[::-1][:k]

print("\n" + "=" * 60)
print("BM25 RESULTS")
print("=" * 60)

for rank, idx in enumerate(
    bm25_indices,
    start=1
):

    print(f"\n--- BM25 Rank {rank} ---")
    print("Chunk index:", idx)
    print("BM25 score:", bm25_scores[idx])
    print(chunks[idx].page_content)


# ==========================================
# 11. Reciprocal Rank Fusion (RRF)
# ==========================================

rrf_scores = {}

rrf_k = 60


# Add FAISS rankings

for rank, idx in enumerate(
    faiss_indices[0],
    start=1
):

    rrf_scores[idx] = rrf_scores.get(idx, 0) + (
        1 / (rrf_k + rank)
    )


# Add BM25 rankings

for rank, idx in enumerate(
    bm25_indices,
    start=1
):

    rrf_scores[idx] = rrf_scores.get(idx, 0) + (
        1 / (rrf_k + rank)
    )


# Sort by combined RRF score

hybrid_results = sorted(
    rrf_scores.items(),
    key=lambda x: x[1],
    reverse=True
)


# ==========================================
# 12. Final Hybrid Results
# ==========================================

print("\n" + "=" * 60)
print("HYBRID RRF RESULTS")
print("=" * 60)

for rank, (idx, score) in enumerate(
    hybrid_results[:k],
    start=1
):

    print(f"\n--- Hybrid Rank {rank} ---")
    print("Chunk index:", idx)
    print("RRF score:", score)
    print(chunks[idx].page_content)
    