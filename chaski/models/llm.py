"""LLM Backend for the chaski-llm application."""

from typing import Any, Dict, Generator, List, Optional

from fastcore.basics import store_attr
from llama_cpp import Llama

from chaski.embeds.engine import EmbeddingsEngine
from chaski.utils.config import Config
from chaski.utils.logging import Logger

logger = Logger(do_setup=False).get_logger(__name__)


def exception_handler(func):
    """Decorator that handles exceptions during response generation."""
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:
            logger.exception(f"Error during {func.__name__}: {exc}")
            raise
    return wrapper


class LLM:
    """LLM backend with optional embedding support."""
    
    def __init__(
        self,
        model_path: str,
        chat_format: Optional[str] = None,
        use_embeddings: bool = False,
        embedding_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initializes the LLM backend.

        Args:
            model_path: The path to the pre-trained `.gguf` model file.
            chat_format: The chat interface for the LLM. Inferred when possible.
            use_embeddings: Whether to enable embeddings.
            embedding_model_kwargs: Keyword arguments for the embedding model.
            **kwargs: Additional keyword arguments for the LLM.
        """
        store_attr()
        self.max_tokens = kwargs.get("max_tokens", Config.MAX_TOKENS)

        self.llm = Llama(model_path=model_path, chat_format=chat_format, **kwargs)
        logger.info(f"Loaded LLM model from {model_path}")

        if use_embeddings:
            self.init_embeddings(embedding_model_kwargs or Config.DEFAULT_EMBEDDINGS)

    def init_embeddings(self, embedding_model_info: Dict[str, Any]):
        """Initializes the embedding model and storage.

        Args:
            embedding_model_info: Configuration dictionary for the embedding model.
        """
        self.embeds = EmbeddingsEngine(embedding_model_info)
        logger.info("Embedding model and storage initialized.")

    @exception_handler
    def embed_and_store(self, text: str, **kwargs):
        """Chunks the given text, then extracts and stores its embeddings.

        Args:
            text: The input text to be embedded and stored.
            **kwargs: Additional keyword arguments for embedding and storing.

        Raises:
            ValueError: If the embedding model is not initialized.
        """
        if not hasattr(self, "embeds"):
            raise ValueError("Embedding model is not initialized.")
        self.embeds.embed_and_store(text, **kwargs)

    @exception_handler
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generates a response for the given prompt.

        Args:
            prompt: The input prompt for generating the response.
            **kwargs: Additional keyword arguments for response generation.

        Returns:
            The generated response text.
        """
        response = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            **kwargs,
        )
        return response["choices"][0]["text"]

    @exception_handler
    def generate_response_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Generates a streaming response for the given prompt.

        Args:
            prompt: The input prompt for generating the response.
            **kwargs: Additional keyword arguments for response generation.

        Yields:
            The generated response text chunks.
        """
        for chunk in self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_tokens,
            stream=True,
            **kwargs,
        ):
            yield chunk["choices"][0]["text"]

    @exception_handler
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generates a chat completion based on the given messages.

        Args:
            messages: The list of messages for generating the chat completion.
            **kwargs: Additional keyword arguments for chat completion.

        Returns:
            The generated chat completion.
        """
        return self.llm.create_chat_completion(
            messages=messages,
            chat_format=self.chat_format,
            **kwargs,
        )

    @exception_handler
    def create_chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generates a streaming chat completion based on the given messages.

        Args:
            messages: The list of messages for generating the chat completion.
            **kwargs: Additional keyword arguments for chat completion.

        Yields:
            The generated chat completion chunks.
        """
        for chunk in self.llm.create_chat_completion(
            messages=messages,
            chat_format=self.chat_format,
            stream=True,
            **kwargs,
        ):
            yield chunk
