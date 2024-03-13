from figma_llm.models.llm_manager import LLMManager
from figma_llm.embeds.db import EmbeddingStorage
from figma_llm.utils.config import Config

# regular mistral instruct format
def prompt_mistral(query, *args, **kwargs):
    """Generates a prompt formatted for RAG with user query and context."""
    return f"{query}"

# Define a function to create a RAG-formatted prompt
def rag_prompt_mistral(query, context):
    """Generates a prompt formatted for RAG with user query and context."""
    return f"{query}\nContext for Instructions: ```{context}```:"


# Initialize the LLM Manager with the desired chat format
print("Initializing the LLM Manager...")
llm_manager = LLMManager(
    model_path=Config.MODEL_PATH,
    use_embeddings=True,  # Enable embedding feature
    embedding_model_info=Config.DEFAULT_EMBEDDINGS  # Use default embedding model settings
)

# Initialize an empty EmbeddingStorage
llm_manager.embedding_storage = EmbeddingStorage(Config.DEFAULT_EMBEDDINGS['file_path'])

# Embed and store a series of sample documents
sample_documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "France is known for its delicious cuisine, including croissants and baguettes.",
    "The Louvre Museum in Paris houses the famous painting, the Mona Lisa.",
]

print("Embedding and storing documents...")
for doc in sample_documents:
    llm_manager.embed_and_store(doc)

# Save the embeddings to a file for persistence
print("Saving the embeddings to a file...")
llm_manager.embedding_storage._save_to_file()

# Load the embeddings from the saved file
print("Loading the embeddings from the file...")
llm_manager.embedding_storage._load_from_file()

# Define a new query to search the embeddings
query = "Tell me about a famous landmark in France."

# Search the embeddings for the top-n most similar to the query
print("Searching for similar documents...")
top_similar = llm_manager.embedding_storage.find_top_n(
    llm_manager.embedding_model.embed(query), n=2
)

# Extract the context from the top similar embeddings
context = "\n".join([text for _, _, text in top_similar])

# Generate a response using the RAG-augmented prompt
print("Generating the response using RAG...")
rag_response = llm_manager.generate_response(rag_prompt_mistral(query, context))

# Compare with a standard response without RAG
print("Generating the response without RAG...")
response = llm_manager.generate_response(prompt_mistral(query))

# Output the results
print(f"Query: {query}")
print(f"Response without RAG:\n{response}")

print(f"Context:\n{context}")
print(f"RAG-Response:\n{rag_response}")
