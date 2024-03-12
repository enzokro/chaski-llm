import argparse
from figma_llm.utils.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Run the LLM from the command line.")
    parser.add_argument("prompt", type=str, help="The input prompt for the LLM.")
    parser.add_argument("--model_path", type=str, default=Config.MODEL_PATH, help="The path to the LLM model file.")
    parser.add_argument("--max_tokens", type=int, default=100, help="The maximum number of tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="The sampling temperature for text generation.")
    args, unknown = parser.parse_known_args()
    return args, unknown