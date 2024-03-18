"""Embedding extraction module for the chaski-llm application."""

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Applies mean pooling to the model output to get sentence embeddings.

    Args:
        model_output: The model output containing token embeddings.
        attention_mask: The attention mask to identify padding tokens.

    Returns:
        The pooled sentence embeddings.
    """
    token_embeddings = model_output[0]  # First element contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class EmbeddingModel:
    """Handles extraction of text embeddings using various libraries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_library: str = "sentence-transformers", **kwargs):
        """Initializes the embedding model based on the specified library and model name.

        Args:
            model_name: Identifier for the model.
            model_library: The library from which to load the model ('sentence-transformers', 'transformers', or 'llama').
            **kwargs: Additional keyword arguments for the model.
        """
        self.model_name = model_name
        self.model_library = model_library
        self.model, self.tokenizer = self._load_model(**kwargs)

    def _load_model(self, **kwargs) -> Tuple[object, Optional[object]]:
        """Loads the model and tokenizer based on the model library.

        Args:
            **kwargs: Additional keyword arguments for the model.

        Returns:
            A tuple containing the loaded model and tokenizer (if applicable).

        Raises:
            ValueError: If the specified model library is not supported.
        """
        if self.model_library == "sentence-transformers":
            return SentenceTransformer(self.model_name), None
        elif self.model_library == "transformers":
            return AutoModel.from_pretrained(self.model_name), AutoTokenizer.from_pretrained(self.model_name)
        elif self.model_library == "llama":
            return Llama(model_path=self.model_name, embedding=True, **kwargs), None
        else:
            raise ValueError(f"Unsupported model library: {self.model_library}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts.

        Args:
            texts: The list of texts to generate embeddings for.

        Returns:
            The list of embeddings, where each embedding is a list of floats.

        Raises:
            ValueError: If the specified model library is not supported for embedding.
        """
        if self.model_library == "sentence-transformers":
            return self.model.encode(texts, convert_to_tensor=False).tolist()
        elif self.model_library == "transformers":
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            return mean_pooling(model_output, encoded_input["attention_mask"]).tolist()
        elif self.model_library == "llama":
            return [self.model.embed(text) for text in texts]
        else:
            raise ValueError(f"Unsupported model library for embedding: {self.model_library}")