import os

class Config:
    """Configuration for the chaski library."""

    # default options for serving the web app
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))

    # NOTE: hardcoded to my model paths
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        "/Users/cck/repos/llama.cpp/models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
    )

    # whether or not to use the embeddings
    USE_EMBEDDINGS = False

    # maximum number of output tokens
    MAX_TOKENS = 256

    # default options for the embedding model
    DEFAULT_EMBEDDINGS = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'model_library': 'sentence-transformers',
        'file_path': 'default_embeddings',
        'file_format': 'npz',
    }