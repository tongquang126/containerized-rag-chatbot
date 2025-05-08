"""
app.py

Flask web application for a RAG-based chatbot, allowing users to interact via a browser.
Serves a simple website with a text input for queries and displays responses.

Usage:
    python src/app.py

Dependencies:
    - flask
    - python-dotenv
    - rag.py (and its dependencies: retriever.py, generator.py, loader.py)
"""

import os
import logging
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from rag import RAGSystem

# Explicitly specify .env file path
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
logger = logging.getLogger(__name__)
logger.debug(f"Attempting to load .env from: {env_path}")

# Load .env file
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.debug(f".env file loaded from: {env_path}")
else:
    logger.warning(f".env file not found at: {env_path}")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize Flask app
# app = Flask(__name__)
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates"))


# Initialize RAG system
try:
    index_path = os.getenv("INDEX_PATH", "./data/faiss_index")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    use_bedrock = os.getenv("USE_BEDROCK", "false").lower() == "true"
    logger.debug(f"OPENAI_API_KEY: {'Set' if openai_api_key else 'Not set'}")
    logger.debug(f"USE_BEDROCK: {use_bedrock}")
    if not openai_api_key and not use_bedrock:
        raise ValueError("OPENAI_API_KEY must be set unless USE_BEDROCK is true")
    rag_system = RAGSystem(index_path)
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
    raise


@app.route("/")
def home():
    """
    Render the main chatbot webpage.

    Returns:
        str: Rendered HTML template.
    """
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    """
    Handle user queries sent via the web interface.

    Returns:
        dict: JSON response with the query result or error message.
    """
    try:
        data = request.get_json()
        query_text = data.get("query", "").strip()
        if not query_text:
            return jsonify({"error": "Query cannot be empty"}), 400

        logger.info(f"Received query: {query_text}")
        answer = rag_system.query(query_text, top_k=3)
        logger.info(f"Generated response for query: {query_text}")
        return jsonify({"response": answer}), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    """
    Start the Flask web server for local testing.

    Example:
        python src/app.py
    """
    try:
        app.run(host="0.0.0.0", port=5001, debug=True)
    except Exception as e:
        logger.error(f"Fatal error starting Flask app: {e}", exc_info=True)
        raise