from fastapi import FastAPI
from pydantic import BaseModel

from backend.rag import WebRAG
from backend.generator import AnswerGenerator

app = FastAPI(title="WebRAG API")

rag = WebRAG()
generator = AnswerGenerator()


class QueryRequest(BaseModel):
    url: str
    question: str


@app.get("/")
def root():
    return {"message": "WebRAG API is running"}


@app.post("/ask")
def ask(request: QueryRequest):

    results = rag.retrieve(
        url=request.url,
        query=request.question,
        k=5
    )

    answer = generator.generate(
        question=request.question,
        results=results
    )

    return {
        "question": request.question,
        "answer": answer["answer"],
        "sources": answer["sources"]
    }     