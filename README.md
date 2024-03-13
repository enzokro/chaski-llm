# Figma LLM 

The Figma LLM Library is a powerful and flexible Python library that provides a seamless interface for working with large language models (LLMs) and embedding-based retrieval.

## Features
- Seamless integration with various LLM providers (e.g., Llama)
- Efficient embedding storage and retrieval using the EmbeddingStorage class
- Flexible embedding extraction with the EmbeddingModel class
- Retrieval-Augmented Generation (RAG) capabilities for enhanced LLM performance
- Modular design for easy customization and extension
- Comprehensive documentation and usage examples

## Project Structure
The project follows a modular structure to ensure maintainability and extensibility. Here's an overview of the main components:  

```bash
figma_llm/
├── embeds/
│   ├── __init__.py
│   ├── db.py
│   └── extract.py
├── models/
│   ├── __init__.py
│   └── llm_manager.py
├── server/
│   ├── __init__.py
│   ├── run.py
│   ├── templates/
│   │   └── index.html
│   └── web_api.py
├── utils/
│   ├── __init__.py
│   ├── args.py
│   ├── config.py
│   ├── distances.py
│   └── txt_chunk.py
├── __init__.py
└── app.py
```

- embeds/: Contains modules for embedding storage and extraction.
- - db.py: Defines the EmbeddingStorage class for storing and retrieving embeddings.
- - extract.py: Defines the EmbeddingModel class for extracting embeddings from text.
- models/: Contains the LLM-related modules.
- - llm_manager.py: Defines the LLMManager class for interacting with the Llama library and managing the RAG workflow.
- server/: Contains the server-related modules.
- - web_api.py: Defines the API endpoints and routes for the web interface.
- run.py: A script for running the web server.
- - templates/: Contains HTML templates for the web interface.
- utils/: Contains utility modules.
- - config.py: Defines the configuration classes for the application.
- - distances.py: Provides distance calculation functions for similarity search.
- - txt_chunk.py: Provides text chunking utilities.
- app.py: The main entry point of the application. 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/figma-llm.git
cd figma-llm
```  

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```  

3. Set up the necessary configuration files (see Configuration).  

### Configuration  
The application uses a Config class to manage configuration settings. Modify the Config class in figma_llm/utils/config.py to adapt the library to your specific needs, such as:

- `MODEL_PATH`: The path to the Llama model file.
- `DEFAULT_EMBEDDINGS`: The default embedding model settings.  

## Running the Application

To run the Figma LLM application, follow these steps:

1. Start the web server:
```bash
python figma_llm/app.py --host localhost --port 5000
```

This will start the server on `http://localhost:5000`.
Open a web browser and navigate to `http://localhost:5000` to access the web interface.


## API Endpoints
The application provides the following API endpoints:

- GET /: Renders the main web interface.  
- POST /: Accepts a prompt parameter in the request form data and generates a response using the LLM.  
- POST /stream: Accepts a prompt parameter in the request form data and generates a response stream using the LLM.  

# Usage

## Initializing the LLM Manager

To get started, create an instance of the LLMManager class, specifying the desired configuration:

```python
from figma_llm.models.llm_manager import LLMManager
from figma_llm.utils.config import Config

llm_manager = LLMManager(
    model_path=Config.MODEL_PATH,
    use_embeddings=True,
    embedding_model_info=Config.DEFAULT_EMBEDDINGS
)
```

## Embedding and Storing Documents
To embed and store documents for later retrieval, use the embed_and_store method:  

```python
documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is a famous landmark in Paris.",
    # ...
]

for doc in documents:
    llm_manager.embed_and_store(doc)

llm_manager.embedding_storage._save_to_file()
```

## Generating Responses
To generate a response from the LLM, use the generate_response method:
  
```python
query = "Tell me about a famous landmark in France."
response = llm_manager.generate_response(query)
```  

## Retrieval-Augmented Generation (RAG)
To leverage the power of RAG, retrieve relevant context from the embedded documents and include it in the prompt:

```python
query = "Tell me about a famous landmark in France."
top_similar = llm_manager.embedding_storage.find_top_n(
    llm_manager.embedding_model.embed(query), n=2
)

context = "\n".join([text for _, _, text in top_similar])
rag_response = llm_manager.generate_response(rag_prompt_mistral(query, context))
```