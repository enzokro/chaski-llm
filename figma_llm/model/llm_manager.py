import logging
from typing import Dict, Any, Generator
from llama_cpp import Llama

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, model_path: str, **kwargs) -> None:
        self.llm = Llama(model_path=model_path, **kwargs)
        logger.info(f"Loaded LLM model from {model_path}")

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
        try:
            output = self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, **kwargs)
            response = output["choices"][0]["text"]
            logger.info(f"Generated response: {response}")
            return response
        except Exception as e:
            logger.exception(f"An error occurred during response generation: {str(e)}")
            raise e

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
        try:
            for chunk in self.llm.create_completion(prompt=prompt, max_tokens=max_tokens, stream=True, **kwargs):
                response_chunk = chunk["choices"][0]["text"]
                logger.info(f"Generated response chunk: {response_chunk}")
                yield response_chunk
        except Exception as e:
            logger.exception(f"An error occurred during response generation: {str(e)}")
            raise e