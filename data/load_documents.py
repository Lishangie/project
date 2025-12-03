"""Document loading utilities."""
import bs4 # type: ignore
from langchain_community.document_loaders import WebBaseLoader  # type: ignore
from langchain_core.documents import Document # type: ignore
from typing import List, Union, Optional


def load_web_documents(
    web_paths: Union[str, tuple],
    bs_kwargs: Optional[dict] = None
) -> List[Document]:
    """
    Load documents from web URLs.
    
    Args:
        web_paths: Single URL string or tuple of URLs
        bs_kwargs: BeautifulSoup kwargs for parsing
        
    Returns:
        List of Document objects
    """
    if bs_kwargs is None:
        bs_kwargs = dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        )
    
    if isinstance(web_paths, str):
        web_paths = (web_paths,)
    
    loader = WebBaseLoader(web_paths=web_paths, bs_kwargs=bs_kwargs)
    docs = loader.load()
    return docs
