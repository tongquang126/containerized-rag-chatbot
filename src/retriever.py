# # src/retriever.py

# """
# Retriever module for loading FAISS index and retrieving relevant documents.
# """

# import os
# import faiss
# import pickle
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# # Initialize global variables for FAISS index and metadata
# index = None
# metadata = None

# def load_faiss_index(faiss_dir: str = "./data/faiss_index"):
#     global index, metadata

#     index_path = os.path.join(faiss_dir, "faiss.index")
#     metadata_path = os.path.join(faiss_dir, "metadata.pkl")

#     if not os.path.exists(index_path):
#         raise FileNotFoundError(f"FAISS index not found at {index_path}")
#     if not os.path.exists(metadata_path):
#         raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

#     index = faiss.read_index(index_path)
#     with open(metadata_path, "rb") as f:
#         metadata = pickle.load(f)
    
#     print(f"Loaded {len(metadata)} documents.")  # Log the number of documents
#     return

# def retrieve(query: str, top_k: int = 3):
#     if top_k <= 0:
#         raise ValueError("top_k must be greater than 0")
#     if index is None or metadata is None:
#         raise RuntimeError("FAISS index and metadata must be loaded with `load_faiss_index()` first.")

#     # Encode the query into embedding vector
#     vector = embedding_model.encode([query])
#     print(f"Query embedding: {vector}")  # Log the query embedding

#     # Search FAISS index
#     distances, indices = index.search(np.array(vector), top_k)
#     print(f"Retrieved indices: {indices}")  # Log the indices

#     # Collect matched metadata
#     results = []
#     for idx in indices[0]:
#         if idx < len(metadata):
#             results.append(metadata[idx])
#     print(f"Retrieved {len(results)} documents.")  # Log the number of retrieved documents
#     return results


# def retrieve_documents(query: str, top_k: int = 3) -> str:
#     load_faiss_index()
#     matches = retrieve(query, top_k)
#     print(f"Matches retrieved: {matches}")  # Log the matches

#     combined_content = "\n\n".join(doc.get("text", "") for doc in matches)
#     print(f"Combined content: {combined_content}")  # Log the combined content
#     return combined_content

# src/retriever.py
"""
retriever.py

This module loads a FAISS index and retrieves relevant document chunks for a given query.
It uses the same SentenceTransformer model as loader.py to embed the query and performs
a similarity search to return the top-k most relevant chunks.

Usage:
    python retriever.py --index-path ./data/faiss_index --query "What is Cloudops tool?" --top-k 3
"""

import argparse
import logging
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load the SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class Retriever:
    """Class to handle loading FAISS index and retrieving relevant document chunks."""
    
    def __init__(self, index_path):
        """
        Initialize the retriever by loading the FAISS index, texts, and metadata.

        Args:
            index_path (str): Directory containing the FAISS index, texts, and metadata.
        """
        self.index, self.texts, self.metadata = self._load_index_and_metadata(index_path)
    
    def _load_index_and_metadata(self, index_path):
        """
        Load the FAISS index, text chunks, and metadata from the specified directory.

        Args:
            index_path (str): Directory containing the FAISS index and metadata.

        Returns:
            tuple: (faiss.Index, List[str], List[dict]) - FAISS index, text chunks, and metadata.
        """
        try:
            # Load FAISS index
            faiss_path = os.path.join(index_path, "faiss.index")
            if not os.path.exists(faiss_path):
                raise FileNotFoundError(f"FAISS index not found at {faiss_path}")
            index = faiss.read_index(faiss_path)
            logger.info(f"Loaded FAISS index from {faiss_path}")

            # Load text chunks
            texts_path = os.path.join(index_path, "texts.pkl")
            if not os.path.exists(texts_path):
                logger.warning(f"Text chunks not found at {texts_path}")
                texts = []
            else:
                with open(texts_path, "rb") as f:
                    texts = pickle.load(f)
                logger.info(f"Loaded {len(texts)} text chunks from {texts_path}")

            # Load metadata
            metadata_path = os.path.join(index_path, "metadata.pkl")
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata not found at {metadata_path}")
                metadata = []
            else:
                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                logger.info(f"Loaded {len(metadata)} metadata entries from {metadata_path}")

            return index, texts, metadata

        except Exception as e:
            logger.error(f"Error loading index or metadata: {e}")
            raise

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k relevant document chunks for a given query.

        Args:
            query (str): Query string to search for.
            top_k (int): Number of top results to return.

        Returns:
            List[dict]: List of results with text, metadata, and similarity score.
        """
        try:
            # Embed the query
            logger.info(f"Embedding query: {query}")
            query_embedding = embedding_model.encode([query], show_progress_bar=False)
            query_embedding = np.array(query_embedding).astype('float32')

            # Search the FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            logger.info(f"Retrieved {len(indices[0])} results from FAISS index")

            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx >= len(self.texts) or idx < 0:
                    logger.warning(f"Invalid index {idx} returned by FAISS")
                    continue

                result = {
                    "text": self.texts[idx] if self.texts else "Text not available",
                    "metadata": self.metadata[idx] if self.metadata else {},
                    "score": 1 / (1 + distance)  # Convert distance to similarity score
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise


def display_results(results):
    """
    Display the query results in a readable format.

    Args:
        results (List[dict]): List of query results with text, metadata, and score.
    """
    if not results:
        logger.info("No results found.")
        return

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Similarity Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}..." if len(result['text']) > 500 else f"Text: {result['text']}")
        print("Metadata:")
        for key, value in result['metadata'].items():
            print(f"  {key}: {value}")
        print("-" * 80)


if __name__ == "__main__":
    """
    Command-line interface to query the FAISS index.

    Example:
        python retriever.py --index-path ./data/faiss_index --query "What is Cloudops tool?" --top-k 3
    """
    parser = argparse.ArgumentParser(description="Retrieve relevant document chunks from a FAISS index")
    parser.add_argument("--index-path", default="./data/faiss_index", 
                        help="Directory containing the FAISS index and metadata")
    parser.add_argument("--query", default="What is Cloudops tool?", 
                        help="Query string to search for")
    parser.add_argument("--top-k", type=int, default=3, 
                        help="Number of top results to retrieve")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")

    args = parser.parse_args()

    # Set the logging level
    logger.setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting retrieval with query: {args.query}")

    try:
        # Initialize the retriever
        retriever = Retriever(args.index_path)

        # Retrieve relevant documents
        results = retriever.retrieve(args.query, args.top_k)

        # Display results
        display_results(results)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())