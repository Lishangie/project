from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.config import *
from src.embedder import get_embeddings
from src.vector_store import create_vectorstore, split_documents
from data import load_documents
from src.retrieval.retriever import retrieve, State
from src.generation.generator import generate

def create_rag_graph(vectorstore, llm):
    """Создать RAG граф"""
    def retrieve_fn(state: State):
        return retrieve(state, vectorstore)
    
    def generate_fn(state: State):
        from src.retrieval.retriever import format_docs
        return generate(state, llm, format_docs)
    
    graph_builder = StateGraph(State).add_sequence([retrieve_fn, generate_fn])
    graph_builder.add_edge(START, "retrieve_fn")
    graph_builder.add_edge("generate_fn", END)
    return graph_builder.compile()

if __name__ == "__main__":
    # Загрузка и обработка
    docs = load_documents("https://lilianweng.github.io/posts/2023-06-23-agent/")
    embeddings = get_embeddings()
    splits = split_documents(docs)
    vectorstore, _ = create_vectorstore(embeddings, splits)
    
    # Создание графа
    # llm = ... # инициализируйте вашу LLM
    # graph = create_rag_graph(vectorstore, llm)
    
    # Запуск
    # response = graph.invoke({"question": "What is Task Decomposition?"})
    # print(response["answer"])
