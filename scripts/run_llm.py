import argparse
import zmq
import logging
from chaski.models.llm import LLMManager
from chaski.utils.config import Config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments for the server."""
    parser = argparse.ArgumentParser(description="Figma LLM Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5001, help="ZMQ port for text generation")
    return parser.parse_args()

def setup_zmq_socket(host: str, port: int):
    """Setup and return a ZMQ socket."""
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{host}:{port}")
    return socket

def main():
    args = parse_arguments()
    llm_manager = LLMManager(model_path=Config.MODEL_PATH)
    socket = setup_zmq_socket(args.host, args.port)
    logger.info(f"ZMQ socket listening on {args.host}:{args.port}")

    while True:
        try:
            data = socket.recv_json()
            response = llm_manager.generate_response(
                prompt=data["prompt"],
                max_tokens=data.get("max_tokens", 100),
                temperature=data.get("temperature", 0.8),
                top_p=data.get("top_p", 0.95)
            )
            socket.send_json({"response": response})
        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)
            socket.send_json({"error": "An error occurred. Please try again."})

if __name__ == "__main__":
    main()
