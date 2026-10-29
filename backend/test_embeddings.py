from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

# Load webpage
loader = WebBaseLoader(url)
documents = loader.load()

# Basic cleaning
text = documents[0].page_content

lines = text.splitlines()

cleaned_lines = []

for line in lines:
    line = line.strip()

    if line:
        cleaned_lines.append(line)

cleaned_text = "\n".join(cleaned_lines)

documents[0].page_content = cleaned_text

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Total chunks:", len(chunks))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
texts = [chunk.page_content for chunk in chunks]

embeddings = model.encode(texts)

print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])