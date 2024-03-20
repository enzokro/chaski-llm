"""Main server module for the chaski-llm application."""

from typing import Optional, Union

from flask import Flask, Response, jsonify, render_template, request

from chaski.models.llm import LLM
from chaski.utils.config import Config
from chaski.utils.logging import Logger

# set up logging
logger = Logger(do_setup=False).get_logger(__name__)


def create_app(
    model_path: str = Config.MODEL_PATH,
    use_embeddings: bool = Config.USE_EMBEDDINGS,
    chat_format: Optional[str] = None,
) -> Flask:
    """Creates and configures the LLM Flask application.

    Args:
        model_path: The path to the pre-trained `.gguf` model file.
        use_embeddings: Whether to enable embeddings.
        chat_format: The chat interface for the LLM. Inferred when possible.

    Returns:
        The configured Flask app.
    """
    app = Flask(__name__)
    llm_manager = LLM(model_path, use_embeddings, chat_format)

    @app.route("/", methods=["GET", "POST"])
    def index() -> Union[str, Response]:
        """Handles the index route and generates responses to prompts.

        Returns:
            The rendered HTML template or a JSON response with an error message.
        """
        if request.method == "POST":
            prompt = request.form.get("prompt", "")
            try:
                # check for an empty prompt
                if not prompt: return jsonify({"error": "Prompt is empty."}), 500

                # Generate a response based on the availability of embeddings
                if llm_manager.use_embeddings and llm_manager.embeds:
                    # Search for top-n most similar embeddings
                    top_n = 3
                    top_similar = llm_manager.embeds.find_top_n(prompt, n=top_n)
                    context = "\n".join([text for _, _, text in top_similar])

                    prompt = f"{prompt}\nContext for Instructions: ```{context}```"
                    logger.info(f"Using RAG prompt: {prompt}")

                # generate a response
                response = llm_manager.generate_response(prompt)

                # append the prompt and response to the history
                app.config["history"].append({"prompt": prompt, "response": response})
                
            except Exception as exc:
                logger.exception(f"Error generating response: {exc}")
                return jsonify({"error": "Failed to generate response."}), 500
        return render_template("index.html", history=app.config["history"])

    @app.route("/stream", methods=["POST"])
    def stream() -> Union[Response, tuple[Response, int]]:
        """Streams responses to prompts on the fly.

        Returns:
            A streaming response with the generated text or a JSON response with an error message.
        """
        prompt = request.form.get("prompt", "")
        try:
            # augment the prompt based on the availability of embeddings
            if llm_manager.use_embeddings and llm_manager.embeds:
                # Search for top-n most similar embeddings
                top_n = 3
                top_similar = llm_manager.embeds.find_top_n(prompt, n=top_n)
                context = "\n".join([text for _, _, text in top_similar])

                prompt = f"{prompt}\nContext for Instructions: ```{context}```"
                logger.info(f"Using RAG prompt: {prompt}")

            # generate a response
            response_generator = llm_manager.generate_response_stream(prompt)

            return Response(response_generator, mimetype="text/event-stream")
        except Exception as exc:
            logger.exception(f"Error in response stream: {exc}")
            return jsonify({"error": "Failed to stream response."}), 500

    @app.route("/upload", methods=["POST"])
    def upload_files():
        uploaded_files = request.files.getlist("files")

        for file in uploaded_files:
            if file.filename.endswith(".txt"):
                text_content = file.read().decode("utf-8")
                llm_manager.embed_and_store(text_content)

        return "Files uploaded successfully", 200

    app.config["history"] = []
    return app


def run_server(
    host: str = Config.HOST,
    port: int = Config.PORT,
    model_path: str = Config.MODEL_PATH,
    use_embeddings: bool = Config.USE_EMBEDDINGS,
    chat_format: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Runs the chaski-llm server.

    Args:
        host: The host address to bind the server to.
        port: The port number to listen on.
        model_path: The path to the pre-trained `.gguf` model file.
        use_embeddings: Whether to enable embeddings.
        chat_format: The chat interface for the LLM. Inferred when possible.
        debug: Whether to run the app in debug mode.
    """
    logger.info(f"Starting chaski-llm server on http://{host}:{port}")
    app = create_app(model_path, chat_format, use_embeddings)
    app.run(host=host, port=port, debug=debug)
