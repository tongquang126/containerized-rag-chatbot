
"""
loader.py

This module loads documents (PDF or TXT) either from a local folder or an S3 bucket,
splits them into manageable text chunks, generates vector embeddings using a sentence-transformer model,
and builds a FAISS index for efficient similarity search.

Usage:
    Run this script from the command line with either 'local' or 's3' mode:
        python src/loader.py --mode local --input ./data/documents --output ./data/faiss_index
        python src/loader.py --mode s3 --input documents/ --bucket my-bucket --output ./data/faiss_index
"""

import os
import io
import boto3
import fitz  # PyMuPDF
import argparse
import logging
from datetime import datetime
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load the SentenceTransformer model once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def split_text(text, chunk_size=500, chunk_overlap=50):
    """
    Split long text into smaller overlapping chunks.

    Args:
        text (str): Full text to be split.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.

    Returns:
        List[str]: List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


def embed_texts(texts, batch_size=32):
    """
    Convert a list of text chunks into vector embeddings.

    Args:
        texts (List[str]): List of text segments.
        batch_size (int): Number of texts to embed at once to manage memory.

    Returns:
        np.ndarray: Array of embedding vectors.
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}/{total_batches} ({len(batch)} texts)")
        batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    return np.vstack(all_embeddings) if all_embeddings else np.array([])


def create_faiss_index(vectors):
    """
    Create a FAISS index from a list of embedding vectors.

    Args:
        vectors (np.ndarray): Matrix of embedding vectors.

    Returns:
        faiss.IndexFlatL2: The built FAISS index object.
    """
    if len(vectors) == 0:
        logger.warning("No vectors to index!")
        # Create an empty index with the correct dimensions
        dim = embedding_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        return index
        
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    return index


def save_index(index, output_path, texts=None, metadata=None):
    """
    Save the FAISS index and optional metadata to the specified directory.

    Args:
        index (faiss.Index): FAISS index object.
        output_path (str): Directory path to store index and metadata.
        texts (List[str], optional): Original text chunks for reference.
        metadata (List[dict], optional): Optional metadata associated with chunks.
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Save the FAISS index
    faiss_path = os.path.join(output_path, "faiss.index")
    faiss.write_index(index, faiss_path)
    logger.info(f"FAISS index saved to {faiss_path}")
    
    # Save the metadata if provided
    if metadata:
        metadata_path = os.path.join(output_path, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Metadata saved to {metadata_path}")
    
    # Optionally save the original texts
    if texts:
        texts_path = os.path.join(output_path, "texts.pkl")
        with open(texts_path, "wb") as f:
            pickle.dump(texts, f)
        logger.info(f"Text chunks saved to {texts_path}")
    
    # Save a simple info file with stats
    info = {
        "created_at": datetime.now().isoformat(),
        "document_count": len(metadata) if metadata else 0,
        "embedding_dim": index.d,
        "index_size": index.ntotal
    }
    
    info_path = os.path.join(output_path, "index_info.pkl")
    with open(info_path, "wb") as f:
        pickle.dump(info, f)


def read_pdf_local(path):
    """
    Read and extract text from a local PDF file.

    Args:
        path (str or Path): Path to a local PDF file.

    Returns:
        str: Full extracted text from the PDF.
    """
    try:
        doc = fitz.open(path)
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF {path}: {e}")
        return ""


def read_txt_local(path):
    """
    Read content from a local TXT file.

    Args:
        path (str or Path): Path to a local .txt file.

    Returns:
        str: File content.
    """
    encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
    
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    
    logger.error(f"Failed to decode {path} with any of the attempted encodings")
    return ""


def read_pdf_s3(s3, bucket, key):
    """
    Read and extract text from a PDF file stored in S3.

    Args:
        s3 (boto3.client): Boto3 S3 client.
        bucket (str): Name of the S3 bucket.
        key (str): Key of the PDF file in S3.

    Returns:
        str: Full extracted text from the PDF.
    """
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        doc = fitz.open(stream=response["Body"].read(), filetype="pdf")
        text = "\n".join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error reading PDF from S3 {bucket}/{key}: {e}")
        return ""


def read_txt_s3(s3, bucket, key):
    """
    Read content from a TXT file stored in S3.

    Args:
        s3 (boto3.client): Boto3 S3 client.
        bucket (str): Name of the S3 bucket.
        key (str): Key of the TXT file in S3.

    Returns:
        str: File content.
    """
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read()
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Failed to decode S3 file {bucket}/{key} with any of the attempted encodings")
        return ""
    
    except Exception as e:
        logger.error(f"Error reading TXT from S3 {bucket}/{key}: {e}")
        return ""


def get_file_metadata(file_path, file_size=None, last_modified=None, is_s3=False):
    """
    Generate metadata for a file.

    Args:
        file_path (str): Path or key to the file.
        file_size (int, optional): Size of the file in bytes.
        last_modified (datetime, optional): Last modified timestamp.
        is_s3 (bool): Whether the file is from S3.

    Returns:
        dict: Metadata dictionary.
    """
    if is_s3:
        source = file_path
        file_name = os.path.basename(file_path)
    else:
        source = str(file_path)
        file_name = file_path.name
        file_size = file_path.stat().st_size if file_size is None else file_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime) if last_modified is None else last_modified
    
    return {
        "source": source,
        "file_name": file_name,
        "file_size": file_size,
        "file_extension": os.path.splitext(file_name)[1].lower(),
        "last_modified": last_modified,
        "processed_at": datetime.now().isoformat()
    }


def process_local(folder_path, output_path, chunk_size=500, chunk_overlap=50):
    """
    Process all PDF and TXT files in a local folder, generate embeddings and save a FAISS index.

    Args:
        folder_path (str): Path to the local folder with documents.
        output_path (str): Path to output FAISS index and metadata.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    """
    all_texts = []
    metadata = []
    file_count = 0
    start_time = time.time()

    try:
        for file_path in Path(folder_path).rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() == ".pdf":
                    logger.info(f"Processing PDF: {file_path}")
                    text = read_pdf_local(file_path)
                elif file_path.suffix.lower() == ".txt":
                    logger.info(f"Processing TXT: {file_path}")
                    text = read_txt_local(file_path)
                else:
                    continue

                if not text.strip():
                    logger.warning(f"Empty or failed to extract text from {file_path}")
                    continue

                file_metadata = get_file_metadata(file_path)
                
                chunks = split_text(text, chunk_size, chunk_overlap)
                logger.info(f"Split into {len(chunks)} chunks")
                
                all_texts.extend(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = file_metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks)
                    })
                    metadata.append(chunk_metadata)
                
                file_count += 1

        if not all_texts:
            logger.warning("No valid text found in any documents!")
            return

        logger.info(f"Found {file_count} files with {len(all_texts)} chunks in total")
        
        logger.info("Generating embeddings...")
        embeddings = embed_texts(all_texts)
        
        logger.info("Creating FAISS index...")
        index = create_faiss_index(embeddings)
        
        logger.info(f"Saving index and metadata to {output_path}...")
        save_index(index, output_path, all_texts, metadata)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        logger.error(traceback.format_exc())


def process_s3(bucket, prefix, output_path, chunk_size=500, chunk_overlap=50):
    """
    Process all PDF and TXT files in a given S3 bucket and prefix, generate embeddings and save FAISS index.

    Args:
        bucket (str): S3 bucket name.
        prefix (str): Prefix in the S3 bucket.
        output_path (str): Path to output FAISS index and metadata.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
    """
    s3 = boto3.client("s3")
    all_texts = []
    metadata = []
    file_count = 0
    start_time = time.time()

    try:
        # Use pagination to handle more than 1000 objects
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in page_iterator:
            if 'Contents' not in page:
                logger.warning(f"No objects found in {bucket}/{prefix}")
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                size = obj['Size']
                last_modified = obj['LastModified']
                
                if key.lower().endswith('.pdf'):
                    logger.info(f"Processing S3 PDF: {bucket}/{key}")
                    text = read_pdf_s3(s3, bucket, key)
                elif key.lower().endswith('.txt'):
                    logger.info(f"Processing S3 TXT: {bucket}/{key}")
                    text = read_txt_s3(s3, bucket, key)
                else:
                    continue
                
                if not text.strip():
                    logger.warning(f"Empty or failed to extract text from {bucket}/{key}")
                    continue
                
                file_metadata = get_file_metadata(
                    key, 
                    file_size=size, 
                    last_modified=last_modified,
                    is_s3=True
                )
                
                chunks = split_text(text, chunk_size, chunk_overlap)
                logger.info(f"Split into {len(chunks)} chunks")
                
                all_texts.extend(chunks)
                
                for i, chunk in enumerate(chunks):
                    chunk_metadata = file_metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "chunk_size": len(chunk),
                        "total_chunks": len(chunks),
                        "s3_bucket": bucket
                    })
                    metadata.append(chunk_metadata)
                
                file_count += 1

        if not all_texts:
            logger.warning("No valid text found in any documents!")
            return

        logger.info(f"Found {file_count} files with {len(all_texts)} chunks in total")
        
        logger.info("Generating embeddings...")
        embeddings = embed_texts(all_texts)
        
        logger.info("Creating FAISS index...")
        index = create_faiss_index(embeddings)
        
        logger.info(f"Saving index and metadata to {output_path}...")
        save_index(index, output_path, all_texts, metadata)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error processing S3 documents: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    """
    Command-line interface to run this script locally or on S3.

    Example:
        python loader.py --mode local --input ./data/documents --output ./data/faiss_index
        python loader.py --mode s3 --bucket my-bucket --input documents/ --output ./data/faiss_index
    """
    parser = argparse.ArgumentParser(description="Process documents and create FAISS index for RAG")
    parser.add_argument("--mode", choices=["local", "s3"], required=True, help="Mode to run: local or s3")
    parser.add_argument("--input", required=True, help="Local path or S3 prefix to input documents")
    parser.add_argument("--bucket", help="S3 bucket name (only required for S3 mode)")
    parser.add_argument("--output", default="data/faiss_index", help="Directory to store FAISS index and metadata")
    parser.add_argument("--chunk-size", type=int, default=500, help="Size of text chunks")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between text chunks")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Set the logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting document processing in {args.mode} mode")
    
    try:
        if args.mode == "local":
            if not os.path.isdir(args.input):
                raise ValueError(f"Input directory doesn't exist: {args.input}")
            process_local(args.input, args.output, args.chunk_size, args.chunk_overlap)
        elif args.mode == "s3":
            if not args.bucket:
                raise ValueError("S3 mode requires --bucket argument")
            process_s3(args.bucket, args.input, args.output, args.chunk_size, args.chunk_overlap)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())