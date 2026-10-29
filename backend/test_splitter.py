from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

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

# Put cleaned text back into a Document
documents[0].page_content = cleaned_text

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Original documents:", len(documents))
print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nFirst chunk length:", len(chunks[0].page_content))

print("\nFirst chunk metadata:")
print(chunks[0].metadata)