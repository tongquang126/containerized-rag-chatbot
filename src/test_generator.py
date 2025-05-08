"""
test_generator.py

Test script for the Generator class in generator.py.
Verifies initialization and answer generation using OpenAI or AWS Bedrock.

Usage:
    python src/test_generator.py

Dependencies:
    - python-dotenv
    - generator.py (and its dependencies: boto3, openai)
"""

import os
import logging
from dotenv import load_dotenv
from generator import Generator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    logger.debug(f".env file loaded from: {env_path}")
else:
    logger.error(f".env file not found at: {env_path}")
    raise FileNotFoundError(f".env file not found at: {env_path}")

def test_generator():
    """
    Test the Generator class by initializing it and generating an answer.

    Uses a sample query and context to verify functionality.
    """
    try:
        # Initialize Generator
        logger.info("Initializing Generator...")
        generator = Generator()
        logger.info("Generator initialized successfully")

        # Sample query and context
        query = "What is Cloudops tool?"
        contexts = [
            {
                "text": "Cloudops is a tool for cloud management, enabling automated deployment, monitoring, and scaling of cloud infrastructure.",
                "metadata": {"source": "test_document.txt"},
                "score": 0.95
            }
        ]

        # Generate answer
        logger.info(f"Generating answer for query: {query}")
        answer = generator.generate_answer(query, contexts)
        logger.info(f"Answer generated: {answer}")

        # Print results
        print("\n=== Generator Test Results ===")
        print(f"Query: {query}")
        print(f"Context: {contexts[0]['text']}")
        print(f"Answer: {answer}")
        print("==============================")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    """
    Run the generator test.

    Example:
        python src/test_generator.py
    """
    try:
        test_generator()
    except Exception as e:
        logger.error(f"Fatal error in test: {e}", exc_info=True)
        raise