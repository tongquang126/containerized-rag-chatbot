# Containerized RAG Chatbot on AWS

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to run on AWS, leveraging containerization for easy deployment. It can ingest and process your internal documents stored in an S3 bucket to provide contextual answers to your queries using large language models.

## ✨ Features

-   **Document Ingestion:** Processes `.txt` and `.pdf` documents stored in an S3 bucket.
-   **Embedding & Indexing:** Embeds text using SentenceTransformer and builds a FAISS index for efficient retrieval.
-   **Contextual Retrieval:** Retrieves the top relevant document chunks based on user queries.
-   **Generative Answers:** Generates contextual answers using OpenAI's GPT-3.5 or GPT-4.
-   **Simple Interface:** Provides a basic chatbot interface powered by Flask.
-   **AWS Ready:** Includes a Dockerfile for easy deployment on AWS ECS.

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/tongquang126/containerized-rag-chatbot.git
    cd containerized-rag-chatbot
    ```

2.  **Set up environment using Pipenv:**
    Make sure you have Pipenv installed:
    ```bash
    pip install pipenv
    ```
    Then install dependencies and activate the virtual environment:
    ```bash
    pipenv install
    pipenv shell
    ```

3.  **Add parameters in `.env`:**
    Create a `.env` file in the root directory and add your configuration:
    ```
    OPENAI_API_KEY=your_openai_key_here
    S3_BUCKET_NAME=containerized-rag-chatbot # Bucket name where stores internal documents and FAISS Index
    S3_DOCUMENTS_PATH=data/documents/  # S3 path to store internal TXT and PDF documents
    S3_INDEX_PATH=data/faiss_index/  # S3 path to store FAISS Index
    LOCAL_FAISS_PATH=./data/faiss_index/ # Local directory where FAISS index will be stored/loaded
    ```

4.  **Upload your documents to the S3\_BUCKET\_NAME:**
    Upload your `.pdf` and `.txt` files into the S3 bucket you specified.

5.  **Build the vector store:**
    Run the following script to process your documents and create the FAISS index:
    ```bash
    python src/loader.py
    ```
    This script loads documents (PDF or TXT) from the specified S3 bucket path, splits them into manageable text chunks, generates vector embeddings using a sentence-transformer model, builds a FAISS index, and saves it to the specified S3 index path.

6.  **Build the Docker image for the chatbox:**
    ```bash
    docker run -p 5001:5001 --env-file .env rag-chatbot
    ```

7.  **Deploy the chatbox to AWS ECS:**
   
## ⚙️ Tech Stack

-   Python
-   LangChain
-   OpenAI API
-   FAISS
-   HuggingFace Transformers
-   Flask
-   Pipenv

## 📄 License

MIT License. Feel free to use, modify, and distribute this project.

## ⚠️ Limitation

This RAG chatbox is designed for static internal documents. Both the documents and the FAISS index are stored on an S3 bucket. The chatbot application downloads the FAISS index to the local disk upon initialization.
