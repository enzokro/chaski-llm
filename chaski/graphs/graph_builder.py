import os
from fastcore.basics import store_attr
from typing import List, Dict, Optional
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM


# Sample generation parameters
MAX_OUTPUT_TOKENS = 4096
TEMP = 0 # make it ~deterministic
TOP_K = 50
TOP_P = 0.95
DO_SAMPLE = True



## INKBOT MODEL SETUP
######################################################################
# set the Inkbot model
# Model Card: `https://huggingface.co/Tostino/Inkbot-13B-8k-0.2`
INKBOT_MODEL_NAME = "Tostino/Inkbot-13B-8k-0.2"

# # NOTE: chat template for Inkbot
# chat_template = "<#meta#>\n- Date: {{ (messages|selectattr('role', 'equalto', 'meta-current_date')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'meta-current_date')|list) else '' }}\n- Task: {{ (messages|selectattr('role', 'equalto', 'meta-task_name')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'meta-task_name')|list) else '' }}\n<#system#>\n{{ (messages|selectattr('role', 'equalto', 'system')|list|last).content|trim if (messages|selectattr('role', 'equalto', 'system')|list) else '' }}\n<#chat#>\n{% for message in messages %}\n{% if message['role'] == 'user' %}\n<#user#>\n{{ message['content']|trim -}}\n{% if not loop.last %}\n\n{% endif %}\n{% elif message['role'] == 'assistant' %}\n<#bot#>\n{{ message['content']|trim -}}\n{% if not loop.last %}\n\n{% endif %}\n{% elif message['role'] == 'user_context' %}\n<#user_context#>\n{{ message['content']|trim -}}\n{% if not loop.last %}\n\n{% endif %}\n{% endif %}\n{% endfor %}\n{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}\n<#bot#>\n{% endif %}"

# sample system message for Inkbot, telling it that it's a generically helpful AI assistant.
SYSTEM_MESSAGE = "You are an AI assistant who will help the user with all their information requests."

# sample user prompt for Inkbot, asking for Knowledge graph construction.
USER_MESSAGE = f"""Your task is to construct a comprehensive Knowledge Graph. 

1. Read and understand the Documents: Please read through the document(s) carefully. As you do, extract the important entities (e.g. key concepts, features, tools related to Figma), their attributes, and relationships between them. The goal is to pull out all and only the information relevant to building an accurate Knowledge Graph. Be comprehensive in capturing all the important information from the document(s), but also be precise in how you represent the entities and relationships.  

2. Create Nodes: Designate each of the essential elements identified earlier as a node with a unique ID using random letters from the greek alphabet. If necessary, add subscripts and superscripts to get more ids. Populate each node with relevant details.  

3. Establish and Describe Edges: Determine the relationships between nodes, forming the edges of your Knowledge Graph. For each edge:
   - Specify the nodes it connects.  
   - Describe the relationship and its direction.  
   - Assign a confidence level (high, medium, low) indicating the certainty of the connection.  

4. Represent All Nodes: Make sure all nodes are included in the edge list.  

After constructing the Knowledge Graph, please output it in its entirety as your final response. The Knowledge Graph should be a structured representation of the key concepts and relationships regarding the Figma design tool, enabling further downstream tasks like question-answering about Figma's features and capabilities.
"""
## /INKBOT MODEL SETUP
######################################################################


def get_today_str():
    """Get today's date in the format 'YYYY-MM-DD'."""
    return datetime.now().strftime("%Y-%m-%d")


class GraphBuilder:
    """Builds a Knowledge Graph from unstructured text using an LLM."""

    def __init__(
            self, 
            model_name: str = INKBOT_MODEL_NAME, 
            system_message: str = SYSTEM_MESSAGE,
            user_message: str = USER_MESSAGE, 
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
            date: str = None, 
            user_message: str= "",
        ) -> str:
        """
        Prepares the prompt for the LLM.

        Args:
            user_context (str): Context information from the document(s).
            user_message (str): User message requesting the Knowledge Graph.
            date Optional[str]: Date to include in the prompt.

        Returns:
            str: Formatted prompt for the LLM.
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

    def _generate_response(
            self, 
            prompt: str,
            max_output_tokens: int = MAX_OUTPUT_TOKENS, 
            **kwargs,
        ) -> str:
        """
        Generates a response from the LLM.

        Args:
            prompt (str): Formatted prompt for the LLM.
            max_output_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional keyword arguments for generation.

        Returns:
            str: Generated response from the LLM.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids,
            # max_length=max_output_tokens, # allow the model to output as much as possible
            num_return_sequences=1,
            temperature=TEMP,
            **kwargs,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def build_graph(self, user_context: str, user_message: str = '', **kwargs) -> str:
        """
        Builds a Knowledge Graph from the given user message and context.
        Note: by default no user_message is provided, so the default Inkbot USER_MESSAGE is used.

        Args:
            user_message (str): User message requesting the Knowledge Graph.
            user_context (str): Context information from the document(s).
            **kwargs: Additional arguments for the `model.generate`.

        Returns:
            str: The extracted Knowledge Graph.
        """
        # optionally override the user message for testing
        if user_message:
            print(f"Overriding user message to: {user_message}")
        prompt = self._prepare_prompt(user_message, user_context)
        response = self._generate_response(prompt, **kwargs)
        return response
    