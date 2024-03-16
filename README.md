# Chaski

Chaski is a powerful and flexible LLM harness built using `llama-cpp-python`. It provides a seamless interface for working with Large Language Models (LLMs) and offers advanced features such as embeddings and knowledge graph generation.

## Key Features

- **LLM Integration**: Chaski seamlessly integrates with LLMs using the `llama-cpp-python` library, allowing you to leverage the power of state-of-the-art language models.  

- **Chat Interface**: Chaski comes with a user-friendly chat interface that allows users to interact with the LLM, input prompts, and view generated responses. The chat interface is built using a web framework and can be easily customized and extended.  

- **Embeddings Support**: Chaski provides functionality to generate, store, and retrieve embeddings from text documents. It utilizes efficient embedding storage and retrieval techniques to enable fast similarity search and retrieval of relevant information.  

- **Retrieval-Augmented Generation (RAG)**: Chaski supports Retrieval-Augmented Generation, a technique that enhances the LLM's response generation by retrieving relevant information from an external knowledge source. RAG enables the LLM to provide more accurate and contextually relevant responses.  

- **Knowledge Graph Generation**: Chaski includes a knowledge graph builder that can extract entities, relationships, and attributes from unstructured text documents and construct a structured knowledge graph. This feature enables users to gain insights and perform complex reasoning tasks based on the extracted knowledge.  

## Getting Started

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/enzokro/chaski-llm.git
    cd chaski-llm
    ```
2. **Install the library and its dependencies**:
    ```bash
    pip install -e .
    pip install -r requirements.txt
    ```

### Project Structure

Below is a high-level overview of Chaski:

```bash
chaski/
├── app.py
├── embeds
│   ├── db.py
│   ├── engine.py
│   └── extract.py
├── graphs
│   └── builder.py
├── models
│   └── llm.py
├── server
│   ├── main.py
│   └── templates
│       └── index.html
└── utils
    ├── config.py
    ├── distances.py
    ├── logging.py
    ├── path_utils.py
    └── txt_chunk.py
```


### Configuration

Customize your setup by adjusting the `Config` class in `chaski_llm/utils/config.py`:

- `MODEL_PATH`: Path to your LLM model. Can also pass in `model_path` at runtime.
- `DEFAULT_EMBEDDINGS`: Default settings for embedding models.


## Running the Chat Interface

To launch the Chaski server:

```bash
python chaski/app.py --host localhost --port 5000
```

Then navigate to http://localhost:5000 in your browser to bring up the chat interface.  


### API Endpoints

The Chaski server currently supports the following endpoints:

- **GET /**: Access the welcoming face of the web interface, designed to be intuitive and user-friendly, guiding you through the process of inputting prompts and viewing responses.

- **POST /**: Submit your `prompt` in the request form data, utilizing the power of LLM to generate insightful and contextually relevant responses.

- **POST /stream**: For continuous interaction with the LLM, this endpoint offers a streaming response capability. It's perfect for applications requiring real-time feedback, dynamically updating the response as more data becomes available.


## Embedding and Storing Documents  

Chaski provides functionality to embed and store text documents for efficient retrieval and similarity search. Here's an example of how to embed and store documents using Chaski:

```python
from chaski.models.llm import LLM
from chaski.utils.config import Config
from chaski.utils.path_utils import get_outputs_dir

# Initialize the LLM
llm = LLM(
    model_path=Config.MODEL_PATH,
    use_embeddings=True,  # Enable embeddings
    embedding_model_info=Config.DEFAULT_EMBEDDINGS,  # Use default embedding settings
)

# Sample documents to embed and store
sample_documents = [
    "The capital of France is Paris.",
    "The Eiffel Tower is a famous landmark in Paris.",
    "France is known for its delicious cuisine, including croissants and baguettes.",
    "The Louvre Museum in Paris houses the famous painting, the Mona Lisa.",
]

# Embed and store the sample documents
for doc in sample_documents:
    llm.embed_and_store(doc)

# Save the embeddings to a file for persistence
out_dir = get_outputs_dir() / 'embeds'
file_path = out_dir / "example_embeddings"
llm.embeds.save_to_file(file_path)
```

## Generating Responses with RAG  

Chaski supports Retrieval-Augmented Generation (RAG) to enhance the LLM's response generation. Here's an example of how to use RAG with Chaski:

```python
from chaski.models.llm import LLM
from chaski.utils.config import Config
from chaski.utils.path_utils import get_outputs_dir

# Initialize the LLM
llm = LLM(
    model_path=Config.MODEL_PATH,
    use_embeddings=True,  # Enable embeddings
    embedding_model_info=Config.DEFAULT_EMBEDDINGS,  # Use default embedding settings
)

# Load the embeddings from a saved file
file_path = get_outputs_dir() / 'embeds' / "example_embeddings"
llm.embeds.load_from_file(file_path)

# Define a new query to search the embeddings
query = "Tell me about a famous landmark in France."

# Search the embeddings for the top-n most similar to the query
top_n = 2
top_similar = llm.embeds.find_top_n(query, n=top_n)

# Extract the context from the top similar embeddings
context = "\n".join([text for _, _, text in top_similar])

# Generate a response using the RAG-augmented prompt
rag_response = llm.generate_response(rag_prompt_mistral(query, context))
```

In this example, we load the previously stored embeddings, search for the top-n most similar embeddings to a given query, extract the relevant context, and generate a response using the RAG-augmented prompt.

## Knowledge Graph Generation 
Chaski includes a knowledge graph builder that can extract entities, relationships, and attributes from unstructured text documents and construct a structured knowledge graph. Here's an example of how to use the knowledge graph builder:

```python
from chaski.graphs.builder import GraphBuilder
from chaski.utils.path_utils import get_outputs_dir, get_project_root

# Set up the graph builder
graph_builder = GraphBuilder()

# Directory with the extracted text documents
documents_dir = get_project_root() / "data/figma_documents"

# Iterate over the documents in the directory
for fid in documents_dir.ls():
    # Read the document
    with open(fid, "r") as file:
        user_context = file.read()

    # Generate the Knowledge Graph
    knowledge_graph = graph_builder.build_graph(
        user_context=user_context,
        user_message="",  # Use the default Graph-building user prompt
    )

    # Save the Knowledge Graph to a file
    out_dir = get_outputs_dir() / 'graphs'
    with open(out_dir / f"{fid.stem}_graph.txt", "w") as file:
        file.write(knowledge_graph)
```

In this example, we set up the knowledge graph builder, iterate over the text documents in a specified directory, generate a knowledge graph for each document, and save the knowledge graph to a file.

Please note that the examples provided assume the necessary dependencies and configurations are in place. Refer to the respective modules and classes for more details on their usage and customization options.

Feel free to explore the Chaski library and leverage its capabilities to build powerful LLM-based applications with embeddings, knowledge graphs, and retrieval-augmented generation!