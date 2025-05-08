"""
test_rag.py

Test script for the RAGSystem class in rag.py.
Verifies initialization, retrieval, and answer generation using a FAISS index and LLM (OpenAI or AWS Bedrock).

Usage:
    python src/test_rag.py

Dependencies:
    - python-dotenv
    - rag.py (and its dependencies: retriever.py, generator.py, faiss-cpu, sentence-transformers, boto3, openai)
"""

import os
import logging
from dotenv import load_dotenv
from rag import RAGSystem

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

def test_rag():
    """
    Test the RAGSystem class by initializing it and processing a sample query.

    Uses a sample query to verify retrieval and generation functionality.
    """
    try:
        # Initialize RAGSystem
        index_path = os.getenv("INDEX_PATH", "./data/faiss_index")
        logger.info(f"Initializing RAGSystem with index_path: {index_path}")
        rag_system = RAGSystem(index_path)
        logger.info("RAGSystem initialized successfully")

        # Sample query
        query = "What is Cloudops tool?"
        top_k = 3

        # Process query
        logger.info(f"Processing query: {query} with top_k={top_k}")
        answer = rag_system.query(query, top_k=top_k)
        logger.info(f"Answer generated: {answer}")

        # Print results
        print("\n=== RAGSystem Test Results ===")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print("==============================")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    """
    Run the RAGSystem test.

    Example:
        python src/test_rag.py
    """
    try:
        test_rag()
    except Exception as e:
        logger.error(f"Fatal error in test: {e}", exc_info=True)
        raise