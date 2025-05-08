"""
rag.py

Retrieval-Augmented Generation (RAG) system integrating retriever and generator modules.
Retrieves relevant document chunks using a FAISS index and generates answers using an LLM
via AWS Bedrock or OpenAI API.

Usage:
    python rag.py --index-path ./data/faiss_index --query "What is Cloudops tool?" --top-k 3

Dependencies:
    - retriever.py
    - generator.py
    - faiss-cpu
    - sentence-transformers
    - boto3 (for Bedrock)
    - openai (for OpenAI API)
"""

import argparse
import logging
from retriever import Retriever
from generator import Generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """Class to integrate retrieval and generation for a RAG system."""
    
    def __init__(self, index_path: str):
        """
        Initialize the RAG system with a retriever and generator.

        Args:
            index_path (str): Directory containing the FAISS index and metadata.
        """
        try:
            self.retriever = Retriever(index_path)
            self.generator = Generator()
            logger.info("Initialized RAG system with retriever and generator")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}", exc_info=True)
            raise

    def query(self, query: str, top_k: int = 3) -> str:
        """
        Process a query by retrieving relevant documents and generating an answer.

        Args:
            query (str): The user question.
            top_k (int): Number of top document chunks to retrieve.

        Returns:
            str: The generated answer or an error message.
        """
        try:
            # Retrieve relevant document chunks
            logger.info(f"Retrieving top {top_k} documents for query: {query}")
            contexts = self.retriever.retrieve(query, top_k)
            
            if not contexts:
                logger.warning("No relevant documents found")
                return "No relevant information found in the documents."

            # Log retrieved contexts
            for i, ctx in enumerate(contexts, 1):
                logger.info(f"Context {i}: Score={ctx['score']:.4f}, Source={ctx.get('metadata', {}).get('source', 'unknown')}")

            # Generate answer using the retrieved contexts
            logger.info("Generating answer from retrieved contexts")
            answer = self.generator.generate_answer(query, contexts)
            return answer

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return f"[RAG Error] {str(e)}"


def main(query: str, index_path: str, top_k: int):
    """
    Main function for standalone testing.

    Args:
        query (str): The user question.
        index_path (str): Directory containing the FAISS index and metadata.
        top_k (int): Number of top document chunks to retrieve.
    """
    rag = RAGSystem(index_path)
    answer = rag.query(query, top_k)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    """
    Command-line interface to test the RAG system.

    Example:
        export USE_BEDROCK=false
        export OPENAI_API_KEY=your-key
        python rag.py --index-path ./data/faiss_index --query "What is Cloudops tool?" --top-k 3
    """
    parser = argparse.ArgumentParser(description="Query a RAG system with retrieval and generation")
    parser.add_argument("--index-path", default="./data/faiss_index", 
                        help="Directory containing the FAISS index and metadata")
    parser.add_argument("--query", default="What is Cloudops tool?", 
                        help="Query string to answer")
    parser.add_argument("--top-k", type=int, default=3, 
                        help="Number of top document chunks to retrieve")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")

    args = parser.parse_args()

    # Set the logging level
    logger.setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting RAG query with: {args.query}")

    try:
        main(args.query, args.index_path, args.top_k)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)