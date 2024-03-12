import logging
from typing import Dict, Any, Generator, List, Optional, Callable
from functools import wraps
from fastcore.basics import *
from llama_cpp import Llama

logger = logging.getLogger(__name__)

def exception_handler(func: Callable) -> Callable:
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
            embedding: bool = False, 
            chat_format: Optional[str] = None, 
            **kwargs
        ):
        store_attr()
        self.llm = Llama(model_path=model_path, embedding=embedding, **kwargs)
        self.chat_format = chat_format
        logger.info(f"Loaded LLM model from {model_path}")


    @exception_handler
    def generate_response(self, prompt: str, max_tokens: int = 100, **kwargs: Any) -> str:
        """
        Generate a response based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            max_tokens (int): The maximum number of tokens to generate. Default is 100.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            str: The generated response.
        """
        output = self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, **kwargs)
        response = output["choices"][0]["text"]
        logger.info(f"Generated response: {response}")
        return response


    @exception_handler
    def generate_response_stream(self, prompt: str, max_tokens: int = 100, **kwargs: Any) -> Generator[str, None, None]:
        """
        Generate a response stream based on the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.
            max_tokens (int): The maximum number of tokens to generate. Default is 100.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Yields:
            str: The generated response chunks.
        """
        for chunk in self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, stream=True, **kwargs):
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
    def embed(self, text: str, **kwargs: Any) -> List[float]:
        """
        Extract embeddings from the given text.

        Args:
            text (str): The input text to extract embeddings from.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            List[float]: The extracted embeddings.
        """
        embeddings = self.llm.embed(text, **kwargs)
        logger.info(f"Extracted embeddings: {embeddings}")
        return embeddings