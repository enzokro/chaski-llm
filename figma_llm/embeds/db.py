import numpy as np
from typing import List, Optional, Dict, Tuple
import hashlib
import os
import logging
from figma_llm.utils.distances import cosine

logger = logging.getLogger(__name__)

def generate_id(text: str) -> str:
    """Generates a unique ID based on the given text using SHA-256 hash."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

class EmbeddingStorage:
    def __init__(self, file_path: str):
        """Simple embedding storage using numpy.
        
        Args:
            file_path (str): The path to the file where embeddings will be saved and loaded.
        """
        self.file_path = file_path
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict] = {}
        self.text_mapping: Dict[str, str] = {}
        
        if os.path.exists(file_path):
            self.load_from_file(file_path)
        
    def add(self, embeddings: List[np.ndarray], metadatas: Optional[List[Dict]] = None, texts: Optional[List[str]] = None) -> List[str]:
        """Add embeddings, their metadata, and corresponding texts to the storage.
        
        Args:
            embeddings (List[np.ndarray]): The list of embeddings to add.
            metadatas (Optional[List[Dict]]): The list of metadata dictionaries corresponding to the embeddings.
            texts (Optional[List[str]]): The list of texts corresponding to the embeddings.
        
        Returns:
            List[str]: The list of generated IDs for the added embeddings.
        """
        ids = []
        for i, embedding in enumerate(embeddings):
            if texts:
                id = generate_id(texts[i])
                self.text_mapping[id] = texts[i]
            else:
                id = generate_id(str(embedding.tolist()))
            self.embeddings[id] = embedding
            if metadatas:
                self.metadata[id] = metadatas[i]
            ids.append(id)
        return ids
    
    def get(self, ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve embeddings by their IDs.
        
        Args:
            ids (List[str]): The list of IDs of the embeddings to retrieve.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary mapping IDs to their corresponding embeddings.
        """
        embeddings = {}
        for id in ids:
            try:
                embeddings[id] = self.embeddings[id]
            except KeyError:
                logger.warning(f"Embedding with ID '{id}' not found in storage.")
        return embeddings
    
    def get_metadata(self, ids: List[str]) -> Dict[str, Dict]:
        """Retrieve metadata by their corresponding embedding IDs.
        
        Args:
            ids (List[str]): The list of IDs of the embeddings whose metadata to retrieve.
        
        Returns:
            Dict[str, Dict]: A dictionary mapping IDs to their corresponding metadata dictionaries.
        """
        return {id: self.metadata.get(id, {}) for id in ids}
    
    def update(self, id: str, embedding: Optional[np.ndarray] = None, metadata: Optional[Dict] = None, text: Optional[str] = None):
        """Update an embedding, its metadata, and corresponding text in the storage.
        
        Args:
            id (str): The ID of the embedding to update.
            embedding (Optional[np.ndarray]): The updated embedding array.
            metadata (Optional[Dict]): The updated metadata dictionary.
            text (Optional[str]): The updated corresponding text.
        """
        if embedding is not None:
            self.embeddings[id] = embedding
        if metadata is not None:
            if id in self.metadata:
                self.metadata[id].update(metadata)
            else:
                self.metadata[id] = metadata
        if text is not None:
            self.text_mapping[id] = text
    
    def delete(self, ids: List[str]):
        """Delete embeddings, their metadata, and corresponding texts from the storage.
        
        Args:
            ids (List[str]): The list of IDs of the embeddings to delete.
        """
        for id in ids:
            try:
                del self.embeddings[id]
                if id in self.metadata:
                    del self.metadata[id]
                if id in self.text_mapping:
                    del self.text_mapping[id]
            except KeyError:
                logger.warning(f"Embedding with ID '{id}' not found in storage. Skipping deletion.")
                
    def save_to_file(self, file_path: Optional[str] = None):
        """Save the embeddings, metadata, and text mapping to a file.
        
        Args:
            file_path (Optional[str]): The path to the file where embeddings, metadata, and text mapping will be saved.
                If not provided, the file path passed during initialization will be used.
        """
        if file_path is None:
            file_path = self.file_path
        try:
            np.savez(file_path, **self.embeddings)
            np.savez(file_path + '_metadata', **self.metadata)
            np.savez(file_path + '_text_mapping', **self.text_mapping)
        except Exception as e:
            logger.error(f"Error occurred while saving embeddings, metadata, and text mapping to file: {str(e)}")
        
    def load_from_file(self, file_path: str):
        """Load embeddings, metadata, and text mapping from a file.
        
        Args:
            file_path (str): The path to the file from which embeddings, metadata, and text mapping will be loaded.
        """
        try:
            loaded_data = np.load(file_path, allow_pickle=True)
            self.embeddings = {key: loaded_data[key] for key in loaded_data.keys()}
            
            metadata_file_path = file_path + '_metadata.npz'
            if os.path.exists(metadata_file_path):
                loaded_metadata = np.load(metadata_file_path, allow_pickle=True)
                self.metadata = {key: loaded_metadata[key].item() for key in loaded_metadata.keys()}
            
            text_mapping_file_path = file_path + '_text_mapping.npz'
            if os.path.exists(text_mapping_file_path):
                loaded_text_mapping = np.load(text_mapping_file_path, allow_pickle=True)
                self.text_mapping = {key: loaded_text_mapping[key].item() for key in loaded_text_mapping.keys()}
        except FileNotFoundError:
            logger.warning(f"Embedding file '{file_path}' not found. Starting with empty storage.")
        except Exception as e:
            logger.error(f"Error occurred while loading embeddings, metadata, and text mapping from file: {str(e)}")
        
    def __del__(self):
        """Save embeddings, metadata, and text mapping to file when the object is destroyed."""
        self.save_to_file()
        
    def upsert(self, embeddings: List[np.ndarray], metadatas: Optional[List[Dict]] = None, texts: Optional[List[str]] = None) -> List[str]:
        """Update or insert embeddings, their metadata, and corresponding texts in the storage.
        
        Args:
            embeddings (List[np.ndarray]): The list of embeddings to upsert.
            metadatas (Optional[List[Dict]]): The list of metadata dictionaries corresponding to the embeddings.
            texts (Optional[List[str]]): The list of texts corresponding to the embeddings.
        
        Returns:
            List[str]: The list of generated IDs for the upserted embeddings.
        """
        ids = []
        for i, embedding in enumerate(embeddings):
            if texts:
                id = generate_id(texts[i])
                self.text_mapping[id] = texts[i]
            else:
                id = generate_id(str(embedding.tolist()))
            self.embeddings[id] = embedding
            if metadatas:
                self.metadata[id] = metadatas[i]
            ids.append(id)
        return ids
    
    def find_top_n_similar(self, query_embedding: np.ndarray, n: int = 5) -> List[Tuple[str, float, str]]:
        """Find the top-n most similar embeddings to the query embedding.
        
        Args:
            query_embedding (np.ndarray): The query embedding to compare against.
            n (int): The number of top similar embeddings to retrieve.
        
        Returns:
            List[Tuple[str, float, str]]: A list of tuples containing the ID, similarity score, and corresponding text
                of the top-n most similar embeddings.
        """
        # Calculate the cosine similarity between the query embedding and all stored embeddings
        similarities = [
            (id, cosine(query_embedding, emb), self.text_mapping.get(id, "")) 
            for id, emb in self.embeddings.items()
        ]
        
        # Sort the embeddings by similarity in ascending order (smaller cosine distance = more similar)
        sorted_similarities = sorted(similarities, key=lambda x: x[1])
        
        # Return the top-n most similar embeddings
        return sorted_similarities[:n]