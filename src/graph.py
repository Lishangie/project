from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .nodes import (
    load_docs,
    split_docs,
    get_embeddings,
    build_vectorstore,
    get_retriever,
    get_prompt,
    get_llm,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_graph():
    docs = load_docs()
    chunks = split_docs(docs)

    embeddings = get_embeddings()
    vectorstore = build_vectorstore(chunks, embeddings)
    retriever = get_retriever(vectorstore)

    prompt = get_prompt()
    llm = get_llm()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain