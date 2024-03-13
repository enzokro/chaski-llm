import os

class Config:
    HOST = os.environ.get("HOST", "0.0.0.0")
    PORT = int(os.environ.get("PORT", 8000))
    MODEL_PATH = os.environ.get(
        "MODEL_PATH",
        "/Users/cck/repos/llama.cpp/models/Mistral-7B-Instruct-v0.2/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
    )

    MAX_TOKENS = 256

    DEFAULT_EMBEDDINGS = {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'model_library': 'sentence-transformers',
        'file_path': 'data/default_embeddings',
        'file_format': 'npz',
    }