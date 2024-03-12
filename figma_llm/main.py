import argparse
import logging
from figma_llm.server.api import run_server

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM App")
    parser.add_argument("--server", action="store_true", help="Run the web server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port number for the server")
    return parser.parse_args()

def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.server:
        logger.info(f"Starting server on {args.host}:{args.port}")
        run_server(args.host, args.port)
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