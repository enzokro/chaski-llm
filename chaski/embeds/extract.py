import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from llama_cpp import Llama

def _mean_pooling(model_output, attention_mask) -> torch.Tensor:
    """
    Applies mean pooling to the model output to get sentence embeddings.

    Args:
        model_output (torch.Tensor): The model output.
        attention_mask (torch.Tensor): The attention mask to identify padding tokens.

    Returns:
        torch.Tensor: The pooled sentence embeddings.
    """
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class EmbeddingModel:
    """Handles extraction of text embeddings using various libraries."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_library: str = "sentence-transformers", **kwargs):
        """
        Initializes the embedding model based on the specified library and model name.
        
        Args:
            model_name (str): Identifier for the model.
            model_library (str): The library from which to load the model ('sentence-transformers', 'transformers', or 'llama').
        """
        self.model_name = model_name
        self.model_library = model_library
        self.model, self.tokenizer = self._load_model()

    def _load_model(self) -> Tuple[Optional[object], Optional[object]]:
        """Loads the model and tokenizer based on the model library."""
        if self.model_library == "sentence-transformers":
            return SentenceTransformer(self.model_name), None
        elif self.model_library == "transformers":
            return AutoModel.from_pretrained(self.model_name), AutoTokenizer.from_pretrained(self.model_name)
        elif self.model_library == "llama":
            return Llama(model_path=self.model_name, embedding=True), None
        else:
            raise ValueError(f"Unsupported model library: {self.model_library}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        if self.model_library == "sentence-transformers":
            return self.model.encode(texts, convert_to_tensor=False).tolist()
        elif self.model_library == "transformers":
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            # TODO: support different poolings
            return _mean_pooling(model_output, encoded_input['attention_mask']).tolist()
        elif self.model_library == "llama":
            return [self.model.embed(text) for text in texts]
        else:
            raise ValueError(f"Unsupported model library for embedding: {self.model_library}")