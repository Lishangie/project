from typing import List
from openai import OpenAI
from langchain_core.embeddings import Embeddings

# OpenAI-совместимый API Ollama
BASE_URL = "http://localhost:11434/v1"
API_KEY = "dummy-key"  # любое непустое

MODEL_NAME = "qllama/multilingual-e5-base:latest"  # как в /api/tags

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

class OllamaEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        resp = client.embeddings.create(
            model=MODEL_NAME,
            input=texts,      # список строк
        )
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        resp = client.embeddings.create(
            model=MODEL_NAME,
            input=[text],
        )
        return resp.data[0].embedding

def get_embeddings() -> Embeddings:
    return OllamaEmbeddings()
