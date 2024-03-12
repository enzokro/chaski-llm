import logging
from typing import Dict, Any, Generator, List, Optional, Callable
from functools import wraps
from fastcore.basics import *
from llama_cpp import Llama
from figma_llm.utils.config import Config
from figma_llm.embeds.extract import EmbeddingModel
from figma_llm.embeds.db import EmbeddingStorage
from figma_llm.utils.txt_chunk import chunk_text 


logger = logging.getLogger(__name__)


def exception_handler(func: Callable) -> Callable:
    """Wraps the function `func` in a try/except block."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"An error occurred during {func.__name__}: {str(e)}")
            raise e
    return wrapper


class LLMManager:
    def __init__(
            self,
            model_path: str, 
            chat_format: Optional[str] = None,
            use_embeddings: bool = False,
            embedding_model_info: Optional[Dict[str, Any]] = Config.DEFAULT_EMBEDDINGS,
            **kwargs,
        ):
        store_attr()

        # create the llm
        self.llm = Llama(
            model_path=model_path,
            chat_format=chat_format,
            **kwargs)
        logger.info(f"Loaded LLM model from {model_path}")

        # initialize the embeddings models
        if use_embeddings:
            self.embedding_model = EmbeddingModel(**embedding_model_info)
            self.embedding_storage = EmbeddingStorage(file_path=embedding_model_info["file_path"])
            logger.info(f"Loaded embedding model and storage: {embedding_model_info}")

        # maximum number of tokens to generate
        self.max_tokens = kwargs.get("max_tokens", Config.MAX_TOKENS)


    @exception_handler
    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            max_tokens (int): The maximum number of tokens to generate. Default is 100.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            str: The generated response.
        """
        output = self.llm.create_completion(prompt=prompt, max_tokens=self.max_tokens, **kwargs)
        response = output["choices"][0]["text"]
        logger.info(f"Generated response: {response}")
        return response


    @exception_handler
    def generate_response_stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Generate a response stream based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            max_tokens (int): The maximum number of tokens to generate. Default is 100.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Yields:
            str: The generated response chunks.
        """
        for chunk in self.llm.create_completion(prompt=prompt, max_tokens=self.max_tokens, stream=True, **kwargs):
            response_chunk = chunk["choices"][0]["text"]
            logger.info(f"Generated response chunk: {response_chunk}")
            yield response_chunk


    @exception_handler
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Generate a chat completion based on the given messages.

        Args:
            messages (List[Dict[str, str]]): The list of messages for the chat completion.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            Dict[str, Any]: The generated chat completion.
        """
        response = self.llm.create_chat_completion(messages=messages, chat_format=self.chat_format, **kwargs)
        logger.info(f"Generated chat completion: {response}")
        return response


    @exception_handler
    def create_chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs: Any) -> Generator[Dict[str, Any], None, None]:
        """
        Generate a chat completion stream based on the given messages.

        Args:
            messages (List[Dict[str, str]]): The list of messages for the chat completion.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Yields:
            Dict[str, Any]: The generated chat completion chunks.
        """
        for chunk in self.llm.create_chat_completion(messages=messages, chat_format=self.chat_format, stream=True, **kwargs):
            logger.info(f"Generated chat completion chunk: {chunk}")
            yield chunk


    @exception_handler
    def embed_and_store(self, text: str, **kwargs: Any):
        """
        Chunk the given text, extract embeddings for each chunk, and store them in EmbeddingStorage.
        
        Args:
            text (str): The input text to extract embeddings from.
            embedding_storage (EmbeddingStorage): The storage system for embeddings and their metadata.
            **kwargs: Additional keyword arguments for embedding extraction.
        """
        if not hasattr(self, 'embedding_model'):
            raise ValueError("Embedding model is not initialized.")

        chunks = chunk_text(text)  # Utilize the provided chunk_text utility
        embeddings = [self.embedding_model.embed(chunk) for chunk in chunks]  # List of embeddings for each chunk

        # Metadata could include the chunk index and the hash of the original text for backtracking
        metadatas = [{"chunk_index": i} for i, _ in enumerate(chunks)]

        # Store embeddings along with their metadata and corresponding chunk of text
        self.embedding_storage.add(embeddings=embeddings, metadatas=metadatas, texts=chunks)