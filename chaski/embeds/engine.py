"""Embeddings engine module for the chaski-llm application."""

from typing import Any, Dict, List, Optional, Tuple

from chaski.embeds.db import EmbeddingStorage
from chaski.embeds.extract import EmbeddingModel
from chaski.utils.config import Config
from chaski.utils.logging import Logger
from chaski.utils.txt_chunk import chunk_text

logger = Logger(do_setup=False).get_logger(__name__)


class EmbeddingsEngine:
    """Handles extraction and storage of text embeddings."""
    
    def __init__(self, embedding_model_info: Optional[Dict[str, Any]] = None, **kwargs):
        model_info = embedding_model_info or Config.DEFAULT_EMBEDDINGS
        self.embedding_model = EmbeddingModel(**model_info)
        self.embedding_storage = EmbeddingStorage(file_path=model_info["file_path"])
        logger.info("Embedding model and storage initialized.")
    
    def embed_and_store(self, text: str, **kwargs):
        """Chunks text, extracts embeddings, and stores them."""
        chunks = chunk_text(text, **kwargs)
        embeddings = [self.embedding_model.embed(chunk) for chunk in chunks]
        metadatas = [{"chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
        self.embedding_storage.add(embeddings=embeddings, metadatas=metadatas)
    
    def find_top_n(self, query: str, n: int = 5, largest_first=False) -> List[Tuple[str, float, str]]:
        """Finds the top-n embeddings most similar to the query."""
        query_embedding = self.embedding_model.embed(query)
        return self.embedding_storage.find_top_n(query_embedding, n, largest_first)
    
    def load_from_file(self, file_path: Optional[str] = None):
        """Loads embeddings and metadata from file."""
        self.embedding_storage._load_from_file(file_path)
    
    def save_to_file(self, file_path: Optional[str] = None):
        """Saves embeddings and metadata to file."""
        self.embedding_storage._save_to_file(file_path)