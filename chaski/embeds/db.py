"""Embedding storage module for the chaski-llm application."""

import hashlib
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from chaski.utils.distances import cosine
from chaski.utils.logging import Logger

logger = Logger(do_setup=False).get_logger(__name__)


def generate_id(text: str) -> str:
    """Generates a unique ID for the given `text`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingStorage:
    """Manages storage and retrieval of text embeddings."""

    def __init__(self, file_path: str, file_format: str = "npz", **kwargs):
        """Initializes the embedding storage.

        Args:
            file_path: The file path for storing the embeddings.
            file_format: The file format for storing the embeddings (default: "npz").
        """
        self.file_path = file_path
        self.file_format = file_format
        self.embeddings: Dict[str, np.ndarray] = {}  # Maps IDs to embeddings
        self.metadata: Dict[str, Dict] = {}          # Maps IDs to metadata
        self._load_from_file()

    def add(self, embeddings: List[np.ndarray], metadatas: Optional[List[Dict]] = None) -> List[str]:
        """Adds new embeddings with optional metadata.

        Args:
            embeddings: The list of embeddings to add.
            metadatas: The list of metadata corresponding to the embeddings (optional).

        Returns:
            The list of generated unique IDs for the added embeddings.
        """
        ids = []
        for i, embedding in enumerate(embeddings):
            metadata = metadatas[i] if metadatas else {}
            text = metadata.get("text", str(embedding.tolist()))
            embedding_id = generate_id(text)
            self.embeddings[embedding_id] = embedding
            self.metadata[embedding_id] = metadata
            ids.append(embedding_id)
        return ids

    def get(self, ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieves embeddings by their IDs.

        Args:
            ids: The list of IDs to retrieve embeddings for.

        Returns:
            A dictionary mapping IDs to their corresponding embeddings.
        """
        return {embedding_id: self.embeddings[embedding_id] for embedding_id in ids if embedding_id in self.embeddings}

    def update(self, embedding_id: str, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None):
        """Updates an existing embedding and its metadata by ID.

        Args:
            embedding_id: The ID of the embedding to update.
            embedding: The updated embedding (optional).
            metadata: The updated metadata (optional).
        """
        if embedding_id in self.embeddings:
            if embedding is not None:
                self.embeddings[embedding_id] = embedding
            if metadata is not None:
                self.metadata[embedding_id].update(metadata)
        else:
            logger.warning(f"Embedding ID '{embedding_id}' not found in storage.")

    def delete(self, ids: List[str]):
        """Removes embeddings and their metadata by IDs.

        Args:
            ids: The list of IDs to remove embeddings for.
        """
        for embedding_id in ids:
            self.embeddings.pop(embedding_id, None)
            self.metadata.pop(embedding_id, None)

    def find_top_n(self, query_embedding: np.ndarray, n: int = 5, largest_first: bool = False) -> List[Tuple[str, float, str]]:
        """Finds the top-n embeddings most similar to the query embedding.

        Args:
            query_embedding: The query embedding to compare against.
            n: The number of top embeddings to retrieve (default: 5).
            largest_first: Whether to sort the results in descending order (default: False).

        Returns:
            A list of tuples containing the ID, distance, and text of the top-n embeddings.
        """
        distances = [
            (embedding_id, cosine(query_embedding, embedding), self.metadata.get(embedding_id, {}).get("text", ""))
            for embedding_id, embedding in self.embeddings.items()
        ]
        return sorted(distances, key=lambda x: x[1], reverse=largest_first)[:n]

    def _load_from_file(self, file_path: str = ""):
        """Loads stored embeddings and metadata from file.

        Args:
            file_path: The file path to load embeddings and metadata from (optional).
        """
        file_path = file_path or self.file_path
        try:
            if os.path.exists(file_path):
                self.embeddings = dict(np.load(f"{file_path}.{self.file_format}", allow_pickle=True))
                self.metadata = dict(np.load(f"{file_path}_metadata.{self.file_format}", allow_pickle=True))
        except Exception as exc:
            logger.exception(f"Failed to load embedding storage: {exc}")

    def _save_to_file(self, file_path: str = ""):
        """Saves the current state of embeddings and metadata to file.

        Args:
            file_path: The file path to save embeddings and metadata to (optional).
        """
        file_path = file_path or self.file_path
        try:
            np.savez(f"{file_path}.{self.file_format}", **self.embeddings)
            np.savez(f"{file_path}_metadata.{self.file_format}", **self.metadata)
        except Exception as exc:
            logger.exception(f"Failed to save embedding storage: {exc}")