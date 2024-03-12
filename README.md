# Figma LLM Python Application

This documentation provides an overview of the Figma LLM Python application, which utilizes the Llama library for text generation and serves as a web interface for user interaction.

## Table of Contents

- [Figma LLM Python Application](#figma-llm-python-application)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
  - [API Endpoints](#api-endpoints)
  - [Text Generation Server](#text-generation-server)
  - [Logging](#logging)
  - [Llama Library Functionality](#llama-library-functionality)
    - [Loading the Llama Model](#loading-the-llama-model)
    - [Text Generation](#text-generation)
    - [Text Generation with Streaming](#text-generation-with-streaming)
    - [Embeddings](#embeddings)
    - [Chat Completion](#chat-completion)
    - [Tokenization and Detokenization](#tokenization-and-detokenization)
    - [Saving and Loading Model States](#saving-and-loading-model-states)

## Project Structure

The project follows a modular structure to ensure maintainability and extensibility. Here's an overview of the main components:

- `figma_llm/`: The main package directory.
 - `factory.py`: Contains the application factory function for creating and configuring the Flask app.
 - `main.py`: The entry point of the application.
 - `model/`: Contains the LLM-related modules.
   - `llm_manager.py`: Defines the `LLMManager` class for interacting with the Llama library.
 - `server/`: Contains the server-related modules.
   - `api.py`: Defines the API endpoints and routes for the web interface.
   - `server_cli.py`: A command-line script for running the text generation server.
 - `utils/`: Contains utility modules.
   - `config.py`: Defines the configuration classes for the application.

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/figma-llm.git
cd figma-llm


2. Install the required dependencies:
pip install -r requirements.txt


3. Set up the necessary configuration files (see [Configuration](#configuration)).

## Configuration

The application uses configuration files to manage various settings. Create a `config.py` file in the `figma_llm/utils/` directory and define the necessary configuration variables, such as:

- `MODEL_PATH`: The path to the Llama model file.
- `LOG_LEVEL`: The log level for the application (e.g., `logging.INFO`, `logging.DEBUG`).
- `LOG_FORMAT`: The format of the log messages.
- `LOG_FILE`: The path to the log file (optional).

## Running the Application

To run the Figma LLM application, follow these steps:

1. Start the Flask development server:
python figma_llm/main.py


This will start the server on `http://localhost:5000`.

2. Open a web browser and navigate to `http://localhost:5000` to access the web interface.

## API Endpoints

The application provides the following API endpoints:

- `GET /`: Renders the main web interface.
- `POST /`: Accepts a `prompt` parameter in the request form data and generates a response using the LLM.
- `POST /stream`: Accepts a `prompt` parameter in the request form data and generates a response stream using the LLM.

## Text Generation Server

The text generation server (`server_cli.py`) is a separate command-line script that listens for text generation requests over a ZMQ socket. It uses the `LLMManager` to generate responses based on the received prompts.

To start the text generation server, run the following command:
python figma_llm/server/server_cli.py --host localhost --port 5001

This will start the server, listening for text generation requests on `localhost:5001`.

Clients can send text generation requests to the server using the ZMQ socket by sending a JSON payload with the necessary parameters (`prompt`, `max_tokens`, `temperature`, `top_p`).

## Logging

The application uses the Python `logging` module for logging purposes. The `configure_logging` function in the `factory.py` file sets up the logging configuration based on the provided settings (`LOG_LEVEL`, `LOG_FORMAT`, `LOG_FILE`).

Log messages are written to the console and optionally to a log file if specified.

## Llama Library Functionality

The Figma LLM application utilizes the Llama library for various functionalities, such as text generation, embeddings, chat completion, tokenization, and more. Here's an overview of the main features:

### Loading the Llama Model

To load the Llama model, create an instance of the `Llama` class:
from llama_cpp import Llama

model_path = "path/to/model"
llm = Llama(model_path=model_path)

### Text Generation

Generate text based on a given prompt using the `create_completion` method:
prompt = "Once upon a time"
response = llm.create_completion(prompt=prompt, max_tokens=50, temperature=0.8, top_p=0.95)
print(response["choices"][0]["text"])


### Text Generation with Streaming

Generate text in a streaming manner using the `create_completion` method with `stream=True`:
prompt = "In a distant galaxy"
for chunk in llm.create_completion(prompt=prompt, max_tokens=100, stream=True):
print(chunk["choices"][0]["text"], end="", flush=True)


### Embeddings

Generate embeddings for a given text using the `create_embedding` method:
text = "The quick brown fox jumps over the lazy dog"
embedding = llm.create_embedding(text)
print(embedding["data"][0]["embedding"])


### Chat Completion

Generate a contextual response based on a list of messages using the `create_chat_completion` method:
messages = [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "What is the capital of France?"},
]
response = llm.create_chat_completion(messages=messages, chat_format="llama-2")
print(response["choices"][0]["message"]["content"])


### Tokenization and Detokenization

Tokenize text into a sequence of tokens and detokenize tokens back into text:
text = "Hello, world!"
tokens = llm.tokenize(text.encode("utf-8"))
print(tokens)

detokenized_text = llm.detokenize(tokens)
print(detokenized_text.decode("utf-8"))


### Saving and Loading Model States

Save and load the state of the Llama model:
state = llm.save_state()

Save the state to a file or database
Load the state
loaded_llm = Llama(model_path=model_path)
loaded_llm.load_state(state)
