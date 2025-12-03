from typing import TypedDict
from langchain_core.documents import Document

class State(TypedDict):
    question: str
    context: list[Document]
    answer: str

def retrieve(state: State, vectorstore):
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
