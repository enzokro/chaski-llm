import fire
import logging
from typing import Any
from figma_llm.models.llm_manager import LLMManager
from figma_llm.utils.args import parse_args
from figma_llm.utils.config import Config

logger = logging.getLogger(__name__)

def run_llm_cli(prompt: str, model_path: str = Config.MODEL_PATH, max_tokens: int = 100, temperature: float = 0.7, timeout: int = 10, **kwargs: Any) -> None:
    """
    Run the LLM from the command line.

    Args:
        prompt (str): The input prompt for the LLM.
        model_path (str): The path to the LLM model file.
        max_tokens (int): The maximum number of tokens to generate. Default is 100.
        temperature (float): The sampling temperature for text generation. Default is 0.7.
        timeout (int): The maximum time in seconds to wait for a response. Default is 10.
        **kwargs: Additional keyword arguments to pass to the LLM.
    """
    try:
        llm_manager = LLMManager(model_path)
        response = llm_manager.generate_response(prompt, max_tokens=max_tokens, temperature=temperature, timeout=timeout, **kwargs)
        print(response)
    except TimeoutError:
        logger.error(f"Response generation timed out after {timeout} seconds")
    except Exception as e:
        logger.exception(f"An error occurred during response generation: {str(e)}")

if __name__ == "__main__":
    args, unknown = parse_args()

    logging.basicConfig(level=logging.INFO)

    # Pass the parsed arguments and additional keyword arguments to Fire
    fire.Fire(lambda **kwargs: run_llm_cli(args.prompt, model_path=args.model_path, max_tokens=args.max_tokens, temperature=args.temperature, timeout=args.timeout, **kwargs))