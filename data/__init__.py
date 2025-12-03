import bs4
from langchain_community.document_loaders import WebBaseLoader

def load_documents(url: str):
    """Загрузить документы с веб-страницы"""
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    return loader.load()
