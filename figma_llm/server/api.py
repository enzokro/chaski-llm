import logging
from typing import Union
from flask import Flask, request, render_template, jsonify, Response
from figma_llm.model.llm_manager import LLMManager
from figma_llm.utils.config import Config

logger = logging.getLogger(__name__)

app = Flask(__name__)

llm_manager = LLMManager(Config.MODEL_PATH)

history = []

@app.route("/", methods=["GET", "POST"])
def index() -> Union[str, Response]:
    if request.method == "POST":
        prompt = request.form["prompt"]
        try:
            response = llm_manager.generate_response(prompt)
            history.append({"prompt": prompt, "response": response})
        except Exception as e:
            logger.exception(f"An error occurred during response generation: {str(e)}")
    return render_template("index.html", history=history)

@app.route("/stream", methods=["POST"])
def stream() -> Response:
    prompt = request.form["prompt"]
    try:
        response_generator = llm_manager.generate_response_stream(prompt)
        return Response(response_generator, mimetype="text/event-stream")
    except Exception as e:
        logger.exception(f"An error occurred during response generation: {str(e)}")
        return jsonify({"error": "An error occurred. Please try again."}), 500

def run_server(host: str, port: int):
    logger.info(f"Server is running on http://{host}:{port}")
    app.run(host=host, port=port)