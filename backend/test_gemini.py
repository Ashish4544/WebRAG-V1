from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("backend/.env")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)

response = llm.invoke("Explain RAG in 2 sentences.")

print("Gemini Response:")
print(response.content)