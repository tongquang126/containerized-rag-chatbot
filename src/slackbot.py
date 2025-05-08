"""
slackbot.py

Slack chatbot integrating a RAG system to answer queries in the System Application Team workspace.
Listens for direct messages or app mentions (e.g., @RAGBot) via the Events API.

Usage:
    python src/slackbot.py

Dependencies:
    - slack-bolt
    - flask
    - python-dotenv
    - rag.py (and its dependencies: retriever.py, generator.py)
"""

import os
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
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
    level=logging.DEBUG,  # Set to DEBUG for troubleshooting
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Initialize Flask app
flask_app = Flask(__name__)

# Initialize Slack Bolt app
try:
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
    slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    logger.debug(f"SLACK_BOT_TOKEN: {'Set' if slack_bot_token else 'Not set'}")
    logger.debug(f"SLACK_SIGNING_SECRET: {'Set' if slack_signing_secret else 'Not set'}")
    logger.debug(f"OPENAI_API_KEY: {'Set' if openai_api_key else 'Not set'}")
    if not slack_bot_token or not slack_signing_secret:
        raise ValueError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET must be set")
    if not slack_bot_token.startswith("xoxb-"):
        raise ValueError("SLACK_BOT_TOKEN is invalid; it must start with 'xoxb-'")
    if not openai_api_key and os.getenv("USE_BEDROCK") != "true":
        logger.warning("OPENAI_API_KEY not set and USE_BEDROCK is not true; RAG may fail")
    app = App(token=slack_bot_token, signing_secret=slack_signing_secret)
    logger.info("Initialized Slack Bolt app")
except Exception as e:
    logger.error(f"Failed to initialize Slack app: {e}", exc_info=True)
    raise

# Initialize RAG system
try:
    index_path = os.getenv("INDEX_PATH", "./data/faiss_index")
    rag_system = RAGSystem(index_path)
    logger.info("RAG system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
    raise


@app.event("app_mention")
def handle_app_mention(event, say):
    """
    Handle app mention events (e.g., @RAGBot What is Cloudops tool?).

    Args:
        event (dict): Slack event data.
        say (callable): Function to send a response.
    """
    try:
        bot_id = event.get("bot_id")
        query = event.get("text", "").replace(f"<@{bot_id}>", "").strip()
        if not query:
            say("Please provide a question (e.g., 'What is Cloudops tool?').")
            return

        logger.info(f"Received app mention query: {query}")
        answer = rag_system.query(query, top_k=3)
        say(answer)
        logger.info(f"Sent response for query: {query}")
    except Exception as e:
        logger.error(f"Error processing app mention: {e}", exc_info=True)
        say(f"[Error] {str(e)}")


@app.event("message")
def handle_message(event, say):
    """
    Handle direct messages to the bot.

    Args:
        event (dict): Slack event data.
        say (callable): Function to send a response.
    """
    try:
        # Only process direct messages (IMs)
        if event.get("channel_type") != "im":
            return

        query = event.get("text", "").strip()
        if not query:
            say("Please provide a question (e.g., 'What is Cloudops tool?').")
            return

        logger.info(f"Received direct message query: {query}")
        answer = rag_system.query(query, top_k=3)
        say(answer)
        logger.info(f"Sent response for query: {query}")
    except Exception as e:
        logger.error(f"Error processing direct message: {e}", exc_info=True)
        say(f"[Error] {str(e)}")


# Flask endpoint for Slack events
handler = SlackRequestHandler(app)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    """
    Handle incoming Slack events, including url_verification.

    Returns:
        Response: Flask response.
    """
    try:
        request_data = request.get_json()
        logger.debug(f"Received request: {request_data}")
        
        # Handle url_verification challenge
        if request_data and request_data.get("type") == "url_verification":
            challenge = request_data.get("challenge")
            if challenge:
                logger.info(f"Responding to url_verification with challenge: {challenge}")
                return jsonify({"challenge": challenge}), 200
            else:
                logger.error("No challenge parameter in url_verification request")
                return jsonify({"error": "Missing challenge parameter"}), 400

        # Delegate other events to Slack Bolt handler
        return handler.handle(request)
    except Exception as e:
        logger.error(f"Error handling Slack events: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    """
    Start the Slack bot in HTTP mode for local testing.

    Example:
        python src/slackbot.py
    """
    try:
        flask_app.run(host="0.0.0.0", port=3000, debug=False)
    except Exception as e:
        logger.error(f"Fatal error starting Slack bot: {e}", exc_info=True)
        raise