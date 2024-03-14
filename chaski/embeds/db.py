import numpy as np
import hashlib
import os
import logging
from typing import List, Dict, Tuple, Optional

from chaski.utils.distances import cosine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_id(text: str) -> str:
    """Generate a unique ID for a given text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class EmbeddingStorage:
    """Manages storage and retrieval of text embeddings."""

    def __init__(self, file_path: str, file_format: str = "npz"):
        self.file_path, self.file_format = file_path, file_format
        self.embeddings = {}  # Maps IDs to embeddings
        self.metadata = {}    # Maps IDs to metadata
        self._load_from_file()

    def add(self, embeddings: List[np.ndarray], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Add new embeddings with optional metadata and texts."""
        ids = []
        for i, embedding in enumerate(embeddings):
            id = generate_id(metadatas[i]['text'] if metadatas else str(embedding.tolist()))
            self.embeddings[id] = embedding
            if metadatas:
                self.metadata[id] = metadatas[i]
            ids.append(id)
        return ids

    def get(self, ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve embeddings by IDs."""
        return {id: self.embeddings.get(id) for id in ids if id in self.embeddings}

    def update(self, id: str, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None, text: Optional[str] = None):
        """Update existing embedding, metadata, and text by ID."""
        if id in self.embeddings or id in self.metadata or id in self.text_mapping:
            if embedding is not None:
                self.embeddings[id] = embedding
            if metadata is not None:
                self.metadata[id].update(metadata)
        else:
            logger.warning(f"ID '{id}' not found in storage.")

    def delete(self, ids: List[str]):
        """Remove embeddings, metadata, and texts by IDs."""
        for id in ids:
            self.embeddings.pop(id, None)
            self.metadata.pop(id, None)

    def find_top_n(self, query_embedding: np.ndarray, n: int = 5, largest_first=False) -> List[Tuple[str, float, str]]:
        """Find top-n embeddings most similar to query."""
        distances = [(id, cosine(query_embedding, emb), self.text_mapping.get(id, "")) for id, emb in self.embeddings.items()]
        return sorted(distances, key=lambda x: x[1], reverse=largest_first)[:n]

    def _load_from_file(self, file_path: str = ''):
        """Load stored embeddings, metadata, and text mappings."""
        file_path = file_path or self.file_path
        try:
            if os.path.exists(file_path):
                self.embeddings = np.load(file_path + '.npz', allow_pickle=True)
                self.metadata = np.load(file_path + '_metadata.npz', allow_pickle=True)
        except Exception as e:
            logger.error(f"Failed to load embedding storage: {e}")

    def _save_to_file(self, file_path: str = ''):
        """Save current state to file."""
        file_path = file_path or self.file_path
        np.savez(file_path + f".{self.file_format}", **self.embeddings)
        np.savez(file_path + f'_metadata.{self.file_format}', **self.metadata)
