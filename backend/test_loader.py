from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

loader = WebBaseLoader(url)

documents = loader.load()

print("Number of documents:", len(documents))

print("\nMetadata:")
print(documents[0].metadata)

print("\nFirst 1000 characters:")
print(documents[0].page_content[:1000])