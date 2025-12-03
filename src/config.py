import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv('.env'))

os.environ["LANGCHAIN_PROJECT"] = "RAG From Scratch: Part 1 (Overview)"

# Добавьте ваши API credentials если они есть в .env
# GIGACHAT_API_BASE_URL = os.environ.get("GIGACHAT_API_BASE_URL")
# GIGACHAT_API_USER = os.environ.get("GIGACHAT_API_USER")
# GIGACHAT_API_PASSWORD = os.environ.get("GIGACHAT_API_PASSWORD")
