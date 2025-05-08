#!/usr/bin/env python3
"""
Test script for the retriever module of the RAG chatbot.
This script tests both the loading of FAISS index and the retrieval functionality.
"""

import os
import sys
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the retriever module
from src.retriever import load_faiss_index, retrieve
# If you need to test the loader, uncomment this
# from src.loader import process_local, process_s3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_test_data(documents_dir: str, index_dir: str) -> None:
    """
    Check if FAISS index exists. If not, create it from the documents directory.
    
    Args:
        documents_dir (str): Path to the directory containing test documents
        index_dir (str): Path to the directory where FAISS index should be stored
    """
    index_path = os.path.join(index_dir, "faiss.index")
    if os.path.exists(index_path):
        logger.info(f"FAISS index found at {index_path}")
        return
    
    # If index doesn't exist, we need to create it
    logger.info(f"FAISS index not found. Creating new index from {documents_dir}")
    
    # Make sure documents directory exists
    if not os.path.exists(documents_dir):
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    # Create parent directory for index if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Import the loader module and process documents
    try:
        from src.loader import process_local
        process_local(documents_dir, index_dir)
        logger.info(f"Successfully created FAISS index at {index_dir}")
    except Exception as e:
        logger.error(f"Failed to create FAISS index: {str(e)}")
        raise


def test_retriever(index_dir: str, queries: List[str], top_k: int = 3) -> None:
    """
    Test the retriever function with a set of queries.
    
    Args:
        index_dir (str): Path to the directory containing the FAISS index
        queries (List[str]): List of query strings to test
        top_k (int): Number of top results to retrieve
    """
    try:
        # Load the FAISS index
        logger.info(f"Loading FAISS index from {index_dir}")
        load_faiss_index(index_dir)
        logger.info("FAISS index loaded successfully")
        
        # Test each query
        for i, query in enumerate(queries, 1):
            logger.info(f"\nQuery {i}: '{query}'")
            results = retrieve(query, top_k=top_k)
            
            # Print results
            logger.info(f"Top {len(results)} results:")
            for j, result in enumerate(results, 1):
                source = result.get("source", "Unknown source")
                chunk_id = result.get("chunk", "Unknown chunk")
                logger.info(f"  {j}. Source: {source}, Chunk: {chunk_id}")
        
        logger.info("\nAll retrieval tests completed successfully.")
    
    except Exception as e:
        logger.error(f"Error during retrieval testing: {str(e)}")
        raise


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Test the retriever functionality of the RAG chatbot")
    parser.add_argument("--index-dir", type=str, default="./data/faiss_index",
                        help="Directory containing the FAISS index")
    parser.add_argument("--docs-dir", type=str, default="./data/documents",
                        help="Directory containing test documents (used only if index doesn't exist)")
    parser.add_argument("--create-index", action="store_true",
                        help="Force creation of a new index even if one exists")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top results to retrieve")
    args = parser.parse_args()
    
    # Sample queries to test the retriever
    test_queries = [
        "what is CloudOps tool?"
    ]
    
    # Create index if needed or requested
    if args.create_index or not os.path.exists(os.path.join(args.index_dir, "faiss.index")):
        setup_test_data(args.docs_dir, args.index_dir)
    
    # Test the retriever
    test_retriever(args.index_dir, test_queries, args.top_k)


if __name__ == "__main__":
    main()