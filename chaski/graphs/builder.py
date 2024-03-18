"""Knowledge Graph construction module using LLMs."""

from datetime import datetime
from typing import Optional, Dict, Any

from fastcore.basics import store_attr
from transformers import AutoTokenizer, AutoModelForCausalLM

from chaski.utils.config import Config


def get_today_str():
    """Get today's date in the format 'YYYY-MM-DD'."""
    return datetime.now().strftime("%Y-%m-%d")


class GraphBuilder:
    """Builds a Knowledge Graph from unstructured text using an LLM."""

    def __init__(
            self, 
            model_name: str = Config.INKBOT_MODEL_NAME, 
            system_message: str = Config.SYSTEM_MESSAGE,
            user_message: str = Config.USER_MESSAGE, 
            generation_kwargs: Optional[Dict[str, Any]] = Config.KG_GENERATION_KWARGS,
            **kwargs,
        ):
        """
        Initializes the GraphBuilder.

        Args:
            model_name (str): Name of the LLM model to use.
            system_message (str): System message to provide context to the LLM.
            **kwargs: Additional keyword arguments for model configuration.
        """
        store_attr()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self) -> AutoTokenizer:
        """Loads the tokenizer for the LLM."""
        return AutoTokenizer.from_pretrained(self.model_name)

    def _load_model(self) -> AutoModelForCausalLM:
        """Loads the LLM model."""
        return AutoModelForCausalLM.from_pretrained(self.model_name)

    def _prepare_prompt(
        self,
        user_context: str,
        user_message: str = "",
        date: Optional[str] = None,
    ) -> str:
        """Prepares the prompt for the LLM.

        The prompt is constructed using a combination of the current date, task name,
        system message, user message, and user context. The prompt is formatted using
        the tokenizer's chat template to ensure compatibility with the LLM.

        Args:
            user_context: Context information from the document(s).
            user_message: User message requesting the Knowledge Graph.
            date: Date to include in the prompt (defaults to current date).

        Returns:
            The formatted prompt for the LLM.
        """
        run_date = date or get_today_str()
        prompt_roles = [
            {"role": "meta-current_date", "content": f"{run_date}"},
            {"role": "meta-task_name", "content": "kg"},
            {"role": "system", "content": self.system_message},
            {"role": "chat", "content": ""},
            {"role": "user", "content": user_message or self.user_message},
            {"role": "user_context", "content": user_context},
        ]
        return self.tokenizer.apply_chat_template(prompt_roles, tokenize=False)

    def _generate_response(self, prompt: str, **kwargs) -> str:
        """Generates a response from the LLM.

        The response is generated using the provided prompt and any additional
        keyword arguments for model generation. The generated output is decoded
        using the tokenizer and returned as a string.

        Args:
            prompt: Formatted prompt for the LLM.
            **kwargs: Additional keyword arguments for model generation.

        Returns:
            The generated response from the LLM.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            num_return_sequences=1,
            **self.generation_kwargs,
            **kwargs,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def build_graph(self, user_context: str, user_message: str = "", **kwargs) -> str:
        """Builds a Knowledge Graph from the given user message and context.

        The Knowledge Graph is constructed by preparing a prompt using the provided
        user message and context, and then generating a response from the LLM. The
        generated response represents the extracted Knowledge Graph.

        Args:
            user_context: Context information from the document(s).
            user_message: User message requesting the Knowledge Graph. If not provided,
                the default user message from the configuration will be used.
            **kwargs: Additional keyword arguments for model generation.

        Returns:
            The extracted Knowledge Graph.
        """
        if user_message:
            print(f"Overriding user message to: {user_message}")
        prompt = self._prepare_prompt(user_context, user_message)
        response = self._generate_response(prompt, **kwargs)
        return response