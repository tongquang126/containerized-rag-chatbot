import os
import logging
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_faiss_index_from_s3():
    """
    Download FAISS index and related files from S3 to local directory.
    Uses environment variables:
        - S3_BUCKET_NAME: The name of the S3 bucket
        - S3_INDEX_PATH: The path/prefix within the S3 bucket (usually 'data/documents/')
        - LOCAL_FAISS_PATH: The local directory where files should be saved
    
    Note: Assumes the actual faiss_index is in a subdirectory 'faiss_index' under S3_INDEX_PATH
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get environment variables
        bucket_name = os.getenv('S3_BUCKET_NAME')
        s3_index_path = os.getenv('S3_INDEX_PATH', 'data/faiss_index/')  # Default if not specified
        local_directory = os.getenv('LOCAL_FAISS_PATH')
        
        # Validate environment variables
        if not bucket_name:
            logger.error("S3_BUCKET_NAME not found in environment variables")
            return False
        if not local_directory:
            logger.error("LOCAL_FAISS_PATH not found in environment variables")
            return False
            
        # Ensure local directory exists
        os.makedirs(local_directory, exist_ok=True)
        logger.info(f"Ensuring local directory exists: {local_directory}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # List of files to download
        files_to_download = [
            "faiss.index",
            "texts.pkl",
            "metadata.pkl"
        ]
        
        # Download each file
        for filename in files_to_download:
            # Construct S3 path - handle potential prefix structure
            s3_path = f"{s3_index_path.rstrip('/')}/{filename}"
            
            local_path = os.path.join(local_directory, filename)
            
            # Check if file already exists locally and log file size/timestamp
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                file_mtime = time.ctime(os.path.getmtime(local_path))
                logger.info(f"File already exists locally: {local_path} (Size: {file_size} bytes, Modified: {file_mtime})")
                continue
            
            # Download the file
            logger.info(f"Downloading {s3_path} from S3 bucket {bucket_name} to {local_path}")
            try:
                s3_client.download_file(bucket_name, s3_path, local_path)
                
                # Verify file was downloaded
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path)
                    logger.info(f"Successfully downloaded {filename} (Size: {file_size} bytes)")
                else:
                    logger.error(f"File download appears to have failed: {local_path} not found")
                    return False
            except ClientError as e:
                logger.error(f"Error downloading file {filename}: {e}")
                return False
        
        return True
    
    except ClientError as e:
        logger.error(f"AWS error downloading from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during S3 download: {e}")
        return False

if __name__ == "__main__":
    # Test the function when run directly
    logger.info("Testing download_faiss_index_from_s3 function")
    success = download_faiss_index_from_s3()
    logger.info(f"Download {'successful' if success else 'failed'}")