"""Server entrypoint for chaski-llm."""

from typing import Optional

import fire

from chaski.server.main import run_server
from chaski.utils.config import Config
from chaski.utils.logging import Logger


# setup logging
logger = Logger(do_setup=False).get_logger(__name__)


def start_server(
    host: str = Config.HOST,
    port: int = Config.PORT,
    model_path: str = Config.MODEL_PATH,
    use_embeddings: bool = Config.USE_EMBEDDINGS,
    chat_format: Optional[str] = None,
) -> None:
    """Starts the chaski web server with the given configs.

    Args:
        host: The host address to bind the server to.
        port: The port number to listen on.
        model_path: The path to the pre-trained `.gguf` model file.
        use_embeddings: Whether to enable embeddings.
        chat_format: The chat interface for the LLM. Inferred when possible.
    """

    # log some startup information
    logger.info(f"Starting chaski server on {host}:{port}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Using embeddings: {use_embeddings}")
    logger.info(f"LLM Chat format: {chat_format}")

    # create and run the application
    run_server(host, port, model_path, use_embeddings, chat_format)


def main():
    """Starts the server."""
    try:
        # run as a fire command-line interface
        fire.Fire(start_server)
    except KeyboardInterrupt:
        # graceful shutdown on keyboard interrupt
        logger.info("Received KeyboardInterrupt, shutting down gracefully...")
    except Exception as exc:
        # log and raise any errors
        logger.exception(f"An unexpected error occurred: {exc}")
        raise


if __name__ == "__main__":
    main()