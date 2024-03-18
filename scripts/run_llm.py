import argparse
import logging
import zmq
from chaski.models.llm import LLM
from chaski.utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parses command line arguments for the server."""
    parser = argparse.ArgumentParser(description="Chaski LLM")
    parser.add_argument("--host", type=str, default="localhost", help="Host for the server.")
    parser.add_argument("--port", type=int, default=5000, help="ZMQ port for text input/generation.")
    return parser.parse_args()


def setup_zmq_socket(host: str, port: int):
    """Sets up and returns a ZMQ REP socket bound to `host:port`."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    return socket


def main():
    """Runs a standalone LLM that listens for prompts and generates responses."""

    # parse cli arguments
    args = parse_arguments()

    # create the default LLM 
    llm_manager = LLM(model_path=Config.MODEL_PATH)

    # setup the socket 
    socket = setup_zmq_socket(args.host, args.port)
    logger.info(f"ZMQ socket listening on {args.host}:{args.port}")

    while True:
        try:
            # wait for data
            data = socket.recv_json()

            # generate and send the response
            response = llm_manager.generate_response(
                prompt=data["prompt"],
                max_tokens=data.get("max_tokens", 100),
                temperature=data.get("temperature", 0.8),
                top_p=data.get("top_p", 0.95)
            )
            socket.send_json({"response": response})

        # log and return exceptions, but do not raise them
        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)
            socket.send_json({"error": "An error occurred. Please try again."})


if __name__ == "__main__":
    main()
