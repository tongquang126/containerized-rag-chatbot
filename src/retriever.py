import sys
import argparse
import logging
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the s3_utils module
from s3_utils import download_faiss_index_from_s3

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
    
    def __init__(self, index_path=None):
        """
        Initialize the retriever by loading the FAISS index, texts, and metadata.
        If the index doesn't exist locally, it downloads from S3 using environment variables.

        Args:
            index_path (str, optional): Override the directory containing the FAISS index.
                                        If None, uses LOCAL_FAISS_PATH from .env file.
        """
        # Use provided index_path or get from environment variable
        self.index_path = index_path or os.getenv('LOCAL_FAISS_PATH')
        
        if not self.index_path:
            raise ValueError("No index path provided. Set LOCAL_FAISS_PATH in .env or provide index_path parameter.")
            
        logger.info(f"Using index path: {self.index_path}")
            
        # Ensure the FAISS index exists locally, if not download from S3
        self.index, self.texts, self.metadata = self._load_index_and_metadata()

    def _ensure_local_directory(self):
        """
        Ensure the local directory for the FAISS index exists.
        Create it if it doesn't exist.
        """
        if not os.path.exists(self.index_path):
            logger.info(f"Creating local directory: {self.index_path}")
            try:
                os.makedirs(self.index_path, exist_ok=True)
                logger.info(f"Successfully created directory: {self.index_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to create directory {self.index_path}: {e}")
                return False
        return True
            
    def _load_index_and_metadata(self):
        """
        Load the FAISS index, text chunks, and metadata from the specified directory.
        If not found locally, download them from S3 using environment variables.

        Returns:
            tuple: (faiss.Index, List[str], List[dict]) - FAISS index, text chunks, and metadata.
        """
        try:
            # Ensure the local directory exists
            if not self._ensure_local_directory():
                raise FileNotFoundError(f"Cannot create or access directory {self.index_path}")
            
            faiss_path = os.path.join(self.index_path, "faiss.index")
            
            # Check if the FAISS index exists
            if not os.path.exists(faiss_path):
                logger.warning(f"FAISS index not found locally at {faiss_path}. Attempting to download from S3.")
                
                # Download from S3 using environment variables
                if download_faiss_index_from_s3():
                    logger.info("Successfully downloaded FAISS index from S3")
                else:
                    raise FileNotFoundError(f"Failed to download FAISS index from S3. Please check your environment variables.")
            
            # Load FAISS index
            index = faiss.read_index(faiss_path)
            logger.info(f"Loaded FAISS index from {faiss_path}")

            # Load text chunks
            texts_path = os.path.join(self.index_path, "texts.pkl")
            if not os.path.exists(texts_path):
                logger.warning(f"Text chunks not found at {texts_path}")
                texts = []
            else:
                with open(texts_path, "rb") as f:
                    texts = pickle.load(f)
                logger.info(f"Loaded {len(texts)} text chunks from {texts_path}")

            # Load metadata
            metadata_path = os.path.join(self.index_path, "metadata.pkl")
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
        python retriever.py --query "What is Cloudops tool?" --top-k 3
    """
    parser = argparse.ArgumentParser(description="Retrieve relevant document chunks from a FAISS index")
    parser.add_argument("--index-path", type=str, default=None, 
                        help="Optional: Directory containing the FAISS index and metadata (overrides LOCAL_FAISS_PATH)")
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
        retriever = Retriever(index_path=args.index_path)

        # Retrieve relevant documents
        results = retriever.retrieve(args.query, args.top_k)

        # Display results
        display_results(results)

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())