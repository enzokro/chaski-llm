import argparse
import logging
from figma_llm.server.api import run_server

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM App")
    parser.add_argument("--server", action="store_true", help="Run the web server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the server")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the LLM model")
    parser.add_argument("--embedding", action="store_true", help="Enable embedding")
    parser.add_argument("--chat_format", type=str, default=None, help="Chat format")
    args, unknown = parser.parse_known_args()
    return args, unknown

def main():
    args, unknown = parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.server:
        logger.info(f"Starting server on {args.host}:{args.port}")
        run_server(
            args.host, 
            args.port, 
            model_path=args.model_path, 
            embedding=args.embedding, 
            chat_format=args.chat_format)
    else:
        logger.info("Please specify an action (--server).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {str(e)}")
        raise e