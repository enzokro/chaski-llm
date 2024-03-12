import argparse
from figma_llm.utils.config import Config

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