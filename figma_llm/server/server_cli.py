import argparse
import zmq
import logging
from figma_llm.models.llm_manager import LLMManager
from figma_llm.utils.config import Config

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Figma LLM Server")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=5001, help="ZMQ port for text generation")
    args = parser.parse_args()

    # Set up the LLMManager
    llm_manager = LLMManager(Config.MODEL_PATH)

    # Set up the ZMQ socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")

    logger.info(f"ZMQ socket listening on port {args.port}")

    while True:
        # Wait for incoming data on the ZMQ socket
        data = socket.recv_json()

        # Process the received data
        prompt = data["prompt"]
        max_tokens = data.get("max_tokens", 100)
        temperature = data.get("temperature", 0.8)
        top_p = data.get("top_p", 0.95)

        try:
            response = llm_manager.generate_response(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            socket.send_json({"response": response})
        except Exception as e:
            logger.exception(f"An error occurred during response generation: {str(e)}")
            socket.send_json({"error": f"An error occurred: {str(e)}.\nPlease try again."})

if __name__ == "__main__":
    main()