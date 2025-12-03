from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return text_splitter.split_documents(docs)

def create_vectorstore(embeddings, splits):
    # Вытаскиваем текст
    texts = [doc.page_content for doc in splits]

    # Отладка: убедиться, что это строки
    print("Sample type:", type(texts[0]))
    print("Sample text snippet:", repr(texts[0][:200]))

    vectorstore = InMemoryVectorStore(embeddings)
    doc_ids = vectorstore.add_texts(texts=texts)
    return vectorstore, doc_ids
