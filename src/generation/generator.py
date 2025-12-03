from langchain_core.messages import HumanMessage

RAG_PROMPT_TEMPLATE = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

def generate(state, llm, format_docs_fn):
    """Сгенерировать ответ на основе контекста"""
    docs_content = format_docs_fn(state["context"])
    rag_prompt = RAG_PROMPT_TEMPLATE.format(
        question=state["question"],
        context=docs_content
    )
    response = llm.invoke([HumanMessage(content=rag_prompt)])
    return {"answer": response.content}
