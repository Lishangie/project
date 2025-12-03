import os
from bs4 import SoupStrainer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

os.environ["USER_AGENT"] = "my-rag/1.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Loader
def load_docs():
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()

# Splitter
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# Embeddings
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# Vector Store
def build_vectorstore(chunks, embeddings):
    return Chroma.from_documents(chunks, embeddings)

# Retriever
def get_retriever(vs):
    return vs.as_retriever()

# Prompt
def get_prompt():
    return ChatPromptTemplate.from_template("""
You are an RAG assistant.

Use the following context to answer the question:

<context>
{context}
</context>

Question: {question}

Answer clearly and concisely based only on the context above.
""")

# LLM
def get_llm():
    return ChatOpenAI(
        model="qwen/qwen3-4b-2507",
        temperature=0,
        base_url="http://127.0.0.1:1234/v1",
        api_key="not-needed",
    )