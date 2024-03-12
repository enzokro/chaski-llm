from typing import List, Any
from llama_cpp import llama_chat_format

"""
Defining custom instruction formats for the LLM.
"""

def custom_instruction_format(
    system_message: str,
    user_prefix: str = "USER:",
    assistant_prefix: str = "ASSISTANT:",
    separator: str = "\n",
):
    def format_custom_instruction(
        messages: List[llama_chat_format.ChatCompletionRequestMessage],
        **kwargs: Any,
    ) -> llama_chat_format.ChatFormatterResponse:
        formatted_messages = []
        for message in messages:
            if message["role"] == "system":
                formatted_messages.append(system_message)
            elif message["role"] == "user":
                formatted_messages.append(f"{user_prefix} {message['content']}")
            elif message["role"] == "assistant":
                formatted_messages.append(f"{assistant_prefix} {message['content']}")
        
        prompt = separator.join(formatted_messages)
        return llama_chat_format.ChatFormatterResponse(prompt=prompt)
    
    return format_custom_instruction


custom_formatter = custom_instruction_format(
    system_message="You are a helpful assistant.",
    user_prefix="Human:",
    assistant_prefix="Assistant:",
    separator="\n\n",
)

llama_chat_format.register_chat_completion_handler("custom_instruction", custom_formatter)