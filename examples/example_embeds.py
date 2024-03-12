from figma_llm.models.llm_manager import LLMManager
from figma_llm.embeds.db import EmbeddingStorage

# Initialize the LLMManager and EmbeddingStorage
################################################################
################################################################
llm_manager = LLMManager(model_path="", embedding=True)
embedding_storage = EmbeddingStorage(file_path="test_embeddings.npz")


# Embed a single string
################################################################
################################################################
text = "This is a sample text to be embedded."
embedding = llm_manager.embed(text)

# Store the embedding
embedding_id = embedding_storage.add(embeddings=[embedding], texts=[text])[0]
print(f"Embedding stored with ID: {embedding_id}")



# Embed several documents
################################################################
################################################################
documents = [
    "This is the first document.",
    "This is the second document.",
    "This is the third document."
]
embeddings = [llm_manager.embed(doc) for doc in documents]

# Store the embeddings
embedding_ids = embedding_storage.add(embeddings=embeddings, texts=documents)
print(f"Embeddings stored with IDs: {embedding_ids}")



# Embed and upsert a new string
################################################################
################################################################
new_text = "This is a new text to be inserted."
new_embedding = llm_manager.embed(new_text)

# Insert the new embedding into the existing database
new_embedding_id = embedding_storage.upsert(embeddings=[new_embedding], texts=[new_text])[0]
print(f"New embedding inserted with ID: {new_embedding_id}")



# Find the top-3 most similar embeddings to a query text
################################################################
################################################################
query_text = "This is a query text similar to the second sample text."
query_embedding = llm_manager.embed(query_text)
top_similar = embedding_storage.find_top_n_similar(query_embedding, n=3)

print(f"Top 3 similar embeddings to '{query_text}':")
for id, similarity, text in top_similar:
    print(f"ID: {id}, Similarity: {similarity}, Text: {text}")