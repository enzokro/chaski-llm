import os


"""Configuration module for the chaski-llm application."""

class Config:
    """Stores configuration values for the chaski-llm application."""

    # General settings
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))

    # LLM settings
    # NOTE: hardcoded to my model paths
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        "/Users/cck/repos/llama.cpp/models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
    )
    MAX_TOKENS = 256
    TEMPERATURE = 0.1
    TOP_P = 0.95

    # Embedding settings
    USE_EMBEDDINGS = True
    DEFAULT_EMBEDDINGS = {
        "model_name": "all-MiniLM-L6-v2",
        "model_library": "sentence-transformers",
        "file_path": "path/to/embeddings",
    }

    # Knowledge Graph settings
    INKBOT_MODEL_NAME = "Tostino/Inkbot-13B-8k-0.2"
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
    KG_GENERATION_KWARGS = {
        "max_output_tokens": 4096,
        "temperature": 0, # make it ~deterministic
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": True,
    }

    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FILE = None
    LOG_ROTATION = True
    LOG_RETENTION = 60
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
