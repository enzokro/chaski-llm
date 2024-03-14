import fire
import logging
from chaski.server.main import run_app
from chaski.utils.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start(
        host: str = Config.HOST, 
        port: int = Config.PORT, 
        model_path: str = Config.MODEL_PATH, 
        use_embeddings: bool = Config.USE_EMBEDDINGS,
        chat_format: str = None,
    ):
    """Starts the web server."""
    
    # Log starting information
    logger.info(f"Starting server on {host}:{port}")
    
    # Launch the server with the provided configurations
    run_app(host, port, model_path=model_path, use_embeddings=use_embeddings, chat_format=chat_format)

if __name__ == "__main__":
    try:
        # Use Fire to handle command-line interfaces
        fire.Fire(start)
    except KeyboardInterrupt:
        # Handle graceful shutdown on keyboard interrupt
        logger.info("Shutting down gracefully...")
    except Exception as e:
        # Log unexpected errors and re-raise
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise e
