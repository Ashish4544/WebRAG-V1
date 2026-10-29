from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# 1. Load webpage
url = "https://m.cricbuzz.com/live-cricket-scores/151532/ind-vs-wi-1st-odi-west-indies-tour-of-india-2026"

loader = WebBaseLoader(url)
documents = loader.load()


# 2. Basic cleaning
text = documents[0].page_content

lines = text.splitlines()

cleaned_lines = []

for line in lines:
    line = line.strip()

    if line:
        cleaned_lines.append(line)

cleaned_text = "\n".join(cleaned_lines)

documents[0].page_content = cleaned_text


# 3. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# 4. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)


# 5. Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings).astype("float32"))

print("Vectors stored in FAISS:", index.ntotal)


# 6. Embed user question
query = "What is the current score and match status?"

query_embedding = model.encode([query])

query_embedding = np.array(query_embedding).astype("float32")


# 7. Search FAISS
k = 3

distances, indices = index.search(query_embedding, k)


# 8. Display results
print("\nQuery:", query)

print("\nTop results:")

for rank, (distance, index_number) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):
    print(f"\n--- Result {rank} ---")
    print("Distance:", distance)
    print("Chunk index:", index_number)
    print(chunks[index_number].page_content)