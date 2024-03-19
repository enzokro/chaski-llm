"""Standalone LLM script that listens for prompts and generates responses."""

import argparse
import zmq

from chaski.models.llm import LLM
from chaski.utils.config import Config
from chaski.utils.logging import Logger

logger = Logger(do_setup=False).get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for the server."""
    parser = argparse.ArgumentParser(description="Chaski LLM")
    parser.add_argument("--host", type=str, default=Config.HOST, help="Host for the server.")
    parser.add_argument("--port", type=int, default=Config.PORT, help="ZMQ port for text input/generation.")
    return parser.parse_args()


def setup_zmq_socket(host: str, port: int) -> zmq.Socket:
    """Sets up and returns a ZMQ REP socket bound to `host:port`."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    return socket


def main() -> None:
    """Runs a standalone LLM that listens for prompts and generates responses."""
    args = parse_arguments()
    llm_manager = LLM(model_path=Config.MODEL_PATH)
    socket = setup_zmq_socket(args.host, args.port)
    logger.info(f"ZMQ socket listening on {args.host}:{args.port}")

    while True:
        try:
            data = socket.recv_json()
            response = llm_manager.generate_response(
                prompt=data["prompt"],
                max_tokens=data.get("max_tokens", Config.MAX_TOKENS),
                temperature=data.get("temperature", Config.TEMPERATURE),
                top_p=data.get("top_p", Config.TOP_P),
            )
            socket.send_json({"response": response})
        except Exception as exc:
            logger.exception(f"Error during response generation: {exc}")
            socket.send_json({"error": "An error occurred. Please try again."})


if __name__ == "__main__":
    main()