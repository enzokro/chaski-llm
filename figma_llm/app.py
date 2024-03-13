import fire
import logging
from figma_llm.server.web_api import server_app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_server(host="0.0.0.0", port=8000, model_path=None, use_embeddings=False, chat_format=None):
    """
    Initialize and start the web server.
    """
    # Log starting information
    logger.info(f"Starting server on {host}:{port}")
    
    # Launch the server with the provided configurations
    server_app(host, port, model_path=model_path, use_embeddings=use_embeddings, chat_format=chat_format)

if __name__ == "__main__":
    try:
        # Use Fire to handle command-line interfaces
        fire.Fire(run_server)
    except KeyboardInterrupt:
        # Handle graceful shutdown on keyboard interrupt
        logger.info("Shutting down gracefully...")
    except Exception as e:
        # Log unexpected errors and re-raise
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise e
