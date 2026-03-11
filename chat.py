import os
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from llm import get_llm, get_embeddings


DB_PATH = "db"


def load_chain(embeddings=None):

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Vector DB not found. Please ingest a PDF first.")

    if embeddings is None:
        embeddings = get_embeddings()

    db = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer ONLY using the context.

Context:
{context}

Question:
{question}

Answer clearly:
""")

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain