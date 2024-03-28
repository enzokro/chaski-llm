#!/usr/bin/env python -B

import sys
sys.dont_write_bytecode = True # to keep things clean
from chaski.models.llm import LLM
from chaski.utils.config import Config
from chaski.utils.path_utils import get_outputs_dir


## Defining our prompts, with and without RAG
################################################
# regular mistral instruct format
def prompt_mistral(query, *args, **kwargs):
    """Generates a simple prompt for the Mistral model."""
    return f"{query}"

# Define a function to create a RAG-formatted prompt
def rag_prompt_mistral(query, context):
    """Generates a prompt formatted for RAG with user query and context."""
    return f"""Using the given context:
{context}

Please answer the following question: {query}
"""
################################################


# Where to save the output embeddings
################################################
out_dir = get_outputs_dir() / 'embeds'
out_dir.mkdir(exist_ok=True)
file_path = out_dir / "example_embeddings"
################################################


## Initialize the LLM
################################################
print("Initializing the LLM Manager...")
llm = LLM(
    model_path=Config.MODEL_PATH,
    use_embeddings=True,  # Enable embeddings
    embedding_model_info=Config.DEFAULT_EMBEDDINGS,  # Use default embedding settings
)
################################################


## Embed and store a series of sample documents
################################################
sample_documents = [
    "Bizcochos de Cayambe are a regional treat in Ecuador.",
    "The bizcochos de Cayambe have a hard, crunchy exterior and flaky interior.",
    "Bizcochos de Cayambe are baked twice, which gives them their signature texture.",
    "Coffee and tea are excellent pairings for bizcochos de Cayambe.",
]

# Embed and store the sample documents
print("Embedding and storing documents...")
for doc in sample_documents:
    llm.embed_and_store(doc)
################################################


# Save and load the embeddings to a file for persistence
################################################
print("Saving the embeddings to a file...")
llm.embeds.save_to_file(file_path)

# Load the embeddings from the saved file
print("Loading the embeddings from the file...")
llm.embeds.load_from_file(file_path)
################################################


# Search the embeddings for the top-n most similar to a query
################################################
# Define a new query to search the embeddings
query = "Tell me about Bizcochos de Cayambe."

# Search the embeddings for the top-n most similar to the query
top_n = 3
print("Searching for the top-{top_n} most similar documents...")
top_similar = llm.embeds.find_top_n(query, n=top_n)

# Extract the context from the top similar embeddings
context = "\n".join([text for _, _, text in top_similar])
print(f"Context for RAG: {context}")
################################################


# Generating with and without RAG
################################################
print("Generating the response without RAG...")
response = llm.generate_response(prompt_mistral(query))

# Generate a response using the RAG-augmented prompt
print("Generating the response using RAG...")
rag_response = llm.generate_response(rag_prompt_mistral(query, context))
################################################


# Output the results with and without RAG
################################################
print(f"Regular Prompt: {prompt_mistral(query)}")
print(f"Response without RAG:\n{response}")

print(f"RAG Prompt: {rag_prompt_mistral(query, context)}")
print(f"Response with RAG:\n{rag_response}")
################################################

