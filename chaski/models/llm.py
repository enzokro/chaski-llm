import logging
from typing import Dict, Any, Generator, List, Optional
from fastcore.basics import store_attr
from llama_cpp import Llama
from chaski.utils.config import Config
from chaski.embeds.engine import EmbeddingsEngine
from chaski.utils.txt_chunk import chunk_text 

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def exception_handler(func):
    """Decorator to handle response generation exceptions."""
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Error during {func.__name__}: {e}")
            raise
    return wrapper

class LLM:
    def __init__(
            self,
            model_path: str,
            chat_format: Optional[str] = None,
            use_embeddings: bool = False,
            embedding_model_kwargs: Optional[Dict[str, Any]] = Config.DEFAULT_EMBEDDINGS,
            **kwargs,
        ):
        store_attr()

        # initialize the llm
        self.llm = Llama(model_path=model_path, chat_format=chat_format, **kwargs)
        logger.info(f"Loaded LLM model from {model_path}")

        self.max_tokens = kwargs.get("max_tokens", Config.MAX_TOKENS)

        # initialize the embeddings engine
        if use_embeddings:
            self.init_embeddings(embedding_model_kwargs) 

    def init_embeddings(self, embedding_model_info: Dict[str, Any]):
        """Initialize embedding model and storage."""
        self.embeds = EmbeddingsEngine(embedding_model_info)
        logger.info("Embedding model and storage initialized.")

    @exception_handler
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate a response for a given prompt."""
        response = self.llm.create_completion(prompt=prompt, max_tokens=self.max_tokens, **kwargs)
        return response["choices"][0]["text"]

    @exception_handler
    def generate_response_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response for a given prompt."""
        for chunk in self.llm.create_completion(prompt=prompt, max_tokens=self.max_tokens, stream=True, **kwargs):
            yield chunk["choices"][0]["text"]

    @exception_handler
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion based on messages."""
        return self.llm.create_chat_completion(messages=messages, chat_format=self.chat_format, **kwargs)

    @exception_handler
    def create_chat_completion_stream(self, messages: List[Dict[str, str]], **kwargs) -> Generator[Dict[str, Any], None, None]:
        """Stream chat completion based on messages."""
        for chunk in self.llm.create_chat_completion(messages=messages, chat_format=self.chat_format, stream=True, **kwargs):
            yield chunk

    @exception_handler
    def embed_and_store(self, text: str, **kwargs):
        """Chunk text, extract, and store embeddings."""
        if not hasattr(self, 'embedding_model'):
            raise ValueError("Embedding model is not initialized.")
        self.embeds.embed_and_store(text, **kwargs)
