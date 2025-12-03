import os
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

from src.config import *
from src.embedder import get_embeddings
from src.vector_store import create_vectorstore, split_documents
from data import load_documents
from src.retrieval.retriever import retrieve, State
from src.generation.generator import generate

from langchain_openai import ChatOpenAI

LLM_BASE_URL = "http://localhost:11434/v1"  # фиксированный URL Ollama

llm = ChatOpenAI(
    model="qwen3:4b",  # Chat-модель из `ollama list`
    base_url=LLM_BASE_URL,
    api_key="dummy-key",
)




def create_rag_graph(vectorstore, llm):
    """Создать RAG граф"""
    def retrieve_fn(state: State):
        return retrieve(state, vectorstore)
    
    def generate_fn(state: State):
        from src.retrieval.retriever import format_docs
        return generate(state, llm, format_docs)

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve_fn)
    graph_builder.add_node("generate", generate_fn)

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    return graph_builder.compile()


if __name__ == "__main__":
    # 1. Загрузка и индексация
    docs = load_documents("https://lilianweng.github.io/posts/2023-06-23-agent/")
    embeddings = get_embeddings()
    splits = split_documents(docs)
    vectorstore, _ = create_vectorstore(embeddings, splits)

    # 2. Создание графа RAG
    graph = create_rag_graph(vectorstore, llm)

    # 3. Вопрос к графу
    question = "О чём говорится в статье?"
    response = graph.invoke({"question": question})

    print("Q:", question)
    print("A:", response["answer"])

