import logging
from typing import Optional, Union
from flask import Flask, request, render_template, jsonify, Response
import fire

from figma_llm.models.llm_manager import LLMManager
from figma_llm.utils.config import Config

# Setup logger for server activities
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Server's conversation history
history = []

def initialize_llm_manager(model_path: str, use_embeddings: bool, chat_format: Optional[str]):
    """Initialize LLM Manager with configured settings."""
    model_path = model_path or Config.MODEL_PATH
    return LLMManager(model_path=model_path, use_embeddings=use_embeddings, chat_format=chat_format)

def server_app(host: str, port: int, model_path: str = "", use_embeddings: bool = False, chat_format: Optional[str] = None):
    """Configure and start the server."""
    llm_manager = initialize_llm_manager(model_path, use_embeddings, chat_format)

    @app.route("/", methods=["GET", "POST"])
    def index() -> Union[str, Response]:
        """Handle index route, generating responses for prompts."""
        if request.method == "POST":
            prompt = request.form.get("prompt")
            try:
                response = llm_manager.generate_response(prompt)
                history.append({"prompt": prompt, "response": response})
            except Exception as e:
                logger.exception("Error generating response: %s", e)
                return jsonify({"error": "Failed to generate response."}), 500
        return render_template("index.html", history=history)

    @app.route("/stream", methods=["POST"])
    def stream() -> Response:
        """Stream responses for prompts."""
        prompt = request.form.get("prompt")
        try:
            response_generator = llm_manager.generate_response_stream(prompt)
            return Response(response_generator, mimetype="text/event-stream")
        except Exception as e:
            logger.exception("Error in response stream: %s", e)
            return jsonify({"error": "Failed to stream response."}), 500

    # Start the Flask server
    logger.info("Server starting on http://%s:%s", host, port)
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    try:
        fire.Fire(server_app)
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated.")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        raise
