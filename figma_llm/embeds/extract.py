# Necessary imports
from typing import List, Union
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Mean Pooling function for transformers
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class EmbeddingModel:
    def __init__(
            self,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
            model_library: str = "sentence-transformers",
            **kwargs,
        ):
        self.model_name = model_name
        self.model_library = model_library
        self.model, self.tokenizer = self.load_model()

    def load_model(self):
        if self.model_library == "sentence-transformers":
            model = SentenceTransformer(self.model_name)
            tokenizer = None  # Not required for sentence-transformers
        elif self.model_library == "transformers":
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(self.model_name)
        elif self.model_library == "llama":
            from llama_cpp import Llama  # Assuming Llama supports a similar interface for embeddings
            model = Llama(model_path=self.model_name, embedding=True)
            tokenizer = None  # Assuming not required for Llama
        else:
            raise ValueError(f"Unsupported model library: {self.model_library}")
        return model, tokenizer

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self.model_library == "sentence-transformers":
            embeddings = self.model.encode(texts, convert_to_tensor=False).tolist()
        elif self.model_library == "transformers":
            encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(sentence_embeddings, p=2, dim=1).tolist()
        elif self.model_library == "llama":
            embeddings = [self.model.embed(text) for text in texts]  # Simplified, adjust based on Llama's actual interface
        else:
            raise ValueError(f"Unsupported model library for embedding: {self.model_library}")
        return embeddings
