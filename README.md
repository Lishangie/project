# RAG System from Scratch

RAG (Retrieval-Augmented Generation) система, реализованная на основе ноутбуков 1-4 из курса "RAG from Scratch".

## Структура проекта

```
project/
├── data/                    # Загрузка документов
│   ├── __init__.py
│   └── load_documents.py   # Утилиты для загрузки документов
├── src/                     # Основной код RAG системы
│   ├── __init__.py
│   ├── config.py           # Конфигурация системы
│   ├── embedder.py         # Embeddings (OpenAI)
│   ├── vector_store.py     # Векторное хранилище (Chroma)
│   ├── retrieval/          # Модуль поиска
│   │   ├── __init__.py
│   │   └── retriever.py    # Retriever для поиска документов
│   └── generation/         # Модуль генерации
│       ├── __init__.py
│       └── generator.py    # RAG chain и генерация ответов
├── main.py                 # Главный файл для запуска
└── pyproject.toml          # Зависимости проекта
```

## Установка

1. Установите зависимости:
```bash
cd project
uv sync
# или
pip install -e .
```

2. Создайте файл `.env` в корне проекта `project/`:
```env
OPENAI_API_KEY=your-openai-api-key-here
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langchain-api-key-here
```

## Использование

### Базовое использование

Запустите систему:
```bash
python main.py
```

Это выполнит:
1. Загрузку документов с веб-страницы
2. Разделение документов на чанки
3. Создание векторного хранилища (Chroma)
4. Создание retriever
5. Создание RAG генератора
6. Ответ на примерный вопрос

### Программное использование

```python
from main import setup_rag_system

# Настройка RAG системы
rag = setup_rag_system()

# Задать вопрос
answer = rag.invoke("What is Task Decomposition?")
print(answer)
```

### Использование отдельных компонентов

```python
from data.load_documents import load_web_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.generator import RAGGenerator

# 1. Загрузка документов
docs = load_web_documents("https://example.com/article")

# 2. Разделение на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

# 3. Создание векторного хранилища
vector_store = VectorStore()
vectorstore = vector_store.create_from_documents(splits)

# 4. Создание retriever
retriever = Retriever(vectorstore=vectorstore, k=4)

# 5. Создание RAG генератора
rag = RAGGenerator(retriever=retriever.get_retriever())

# 6. Генерация ответа
answer = rag.invoke("Your question here")
```

## Компоненты системы

### 1. Загрузка документов (`data/load_documents.py`)
- Загрузка документов с веб-страниц через `WebBaseLoader`
- Поддержка BeautifulSoup для парсинга HTML

### 2. Embeddings (`src/embedder.py`)
- Обертка над OpenAI Embeddings
- Методы для embedding запросов и документов

### 3. Векторное хранилище (`src/vector_store.py`)
- Использует Chroma для хранения векторов
- Поддержка персистентности (сохранение на диск)
- Создание нового хранилища или загрузка существующего

### 4. Retriever (`src/retrieval/retriever.py`)
- Поиск релевантных документов по запросу
- Настраиваемое количество возвращаемых документов (k)

### 5. Генератор (`src/generation/generator.py`)
- RAG chain для генерации ответов
- Использует LangChain Hub для промптов или кастомные шаблоны
- Интеграция с OpenAI Chat API

## Конфигурация

Настройки можно изменить в `src/config.py`:

- `CHUNK_SIZE`: Размер чанков при разделении текста (по умолчанию: 1000)
- `CHUNK_OVERLAP`: Перекрытие между чанками (по умолчанию: 200)
- `RETRIEVER_K`: Количество документов для поиска (по умолчанию: 4)
- `LLM_MODEL_NAME`: Модель LLM (по умолчанию: "gpt-3.5-turbo")
- `LLM_TEMPERATURE`: Температура LLM (по умолчанию: 0.0)

## Зависимости

Основные зависимости:
- `langchain` - фреймворк для работы с LLM
- `langchain-openai` - интеграция с OpenAI
- `langchain-community` - дополнительные компоненты
- `chromadb` - векторное хранилище
- `tiktoken` - подсчет токенов
- `langchainhub` - промпты из Hub

Полный список зависимостей см. в `pyproject.toml`.

## Примечания

- Векторное хранилище сохраняется в директории `./chroma_db` (можно изменить через `VECTOR_STORE_PERSIST_DIRECTORY` в `.env`)
- При первом запуске создается новое хранилище, при последующих запусках загружается существующее
- Для пересоздания хранилища удалите директорию `chroma_db` или используйте `recreate_vectorstore=True` в `setup_rag_system()`
