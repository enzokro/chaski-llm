import logging
from typing import List, Dict, Optional, Tuple, Any
from chaski.embeds.db import EmbeddingStorage
from chaski.embeds.extract import EmbeddingModel
from chaski.utils.config import Config
from chaski.utils.txt_chunk import chunk_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingsEngine: 
    def __init__(self, embedding_model_info: Optional[Dict[str, Any]] = Config.DEFAULT_EMBEDDINGS, **kwargs):
        """Handles extraction and storage of text embeddings."""
        self.embedding_model = EmbeddingModel(**embedding_model_info)
        self.embedding_storage = EmbeddingStorage(file_path=embedding_model_info["file_path"])
        logger.info("Embedding model and storage initialized.")
    
    def embed_and_store(self, text: str, **kwargs):
        """Chunk text, extract, and store embeddings for `text`."""
        chunks = self.chunk_text(text)
        embeddings = [self.embedding_model.embed(chunk) for chunk in chunks]
        metadatas = [{"chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
        self.embedding_storage.add(embeddings=embeddings, metadatas=metadatas)

    def chunk_text(self, text: str, **kwargs):
        """Small wrapper for text chunking."""
        return chunk_text(text, **kwargs)
    
    def find_top_n(self, query: str, n: int = 5, largest_first=False) -> List[Tuple[str, float, str]]:
        """Find top-n embeddings most similar to the query."""
        query_embedding = self.embedding_model.embed(query)
        return self.embedding_storage.find_top_n(query_embedding, n, largest_first)
    
    def load_from_file(self, file_path: Optional[str] = None):
        """Load embeddings and metadata from file."""
        self.embedding_storage._load_from_file(file_path)
    
    def save_to_file(self, file_path: Optional[str] = None):
        """Save embeddings and metadata to file."""
        self.embedding_storage._save_to_file(file_path)