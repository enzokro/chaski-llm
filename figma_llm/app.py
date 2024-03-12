import fire
import logging
from figma_llm.server.api import run_server

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def start_server(
        server: bool = False, 
        host: str= "0.0.0.0",
        port: int = 8000, 
        model_path: str = None, 
        use_embeddings: bool = False, 
        chat_format: str = None, 
        # **kwargs,
    ):
    """
    Starts the web server with specified parameters and any additional arbitrary arguments.
    
    Args:
        server (bool): Flag to run the web server.
        host (str): Host address for the server.
        port (int): Port number for the server.
        model_path (str): Path to the LLM model.
        use_embeddings (bool): Enable embedding functionality.
        chat_format (str): Specify the chat format.
        **kwargs: Arbitrary keyword arguments.
    """
    
    logger.info(f"Starting server on {host}:{port} with extra args: {kwargs}")
    run_server(
        host, 
        port, 
        model_path=model_path,
        use_embeddings=use_embeddings,
        chat_format=chat_format,
        # **kwargs,
        )

if __name__ == "__main__":
    try:
        fire.Fire(start_server)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise e