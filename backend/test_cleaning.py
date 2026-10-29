from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader

load_dotenv()

url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

loader = WebBaseLoader(url)
documents = loader.load()

text = documents[0].page_content

# Basic cleaning
lines = text.splitlines()

cleaned_lines = []

for line in lines:
    line = line.strip()

    if line:
        cleaned_lines.append(line)

cleaned_text = "\n".join(cleaned_lines)

print("Original characters:", len(text))
print("Cleaned characters:", len(cleaned_text))

print("\nFirst 2000 characters after cleaning:")
print(cleaned_text[:2000])