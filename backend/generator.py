from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("backend/.env")


class AnswerGenerator:

    def __init__(self):

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful assistant that answers questions
                about webpages.

                Answer using only the provided webpage context.
                If the context does not contain the answer,
                say you couldn't find the answer on this page.

                Keep answers clear and concise."""
            ),
            (
                "human",
                """Question: {question}

                Webpage context:
                {context}"""
            )
        ])

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            temperature=0
        )

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question, results):

        if not results:
            return {
                "answer": "I couldn't find relevant information on this page.",
                "sources": []
            }

        context = "\n\n".join(
            result["text"]
            for result in results
        )

        answer = self.chain.invoke({
            "question": question,
            "context": context
        })

        sources = list(dict.fromkeys(
            result["source"]
            for result in results
        ))

        return {
            "answer": answer,
            "sources": sources
        }