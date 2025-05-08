"""
generator.py

Generator module for querying a large language model (LLM) via AWS Bedrock or OpenAI API.
Supports switching between the two based on the USE_BEDROCK environment variable.

Usage:
    python generator.py --query "What is Cloudops tool?" --context "Cloudops is a tool for cloud management..."

Dependencies:
    - boto3 (for AWS Bedrock)
    - openai (for OpenAI API)
"""

import os
import json
import boto3
import logging
import argparse
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize LLM client based on USE_BEDROCK environment variable
USE_BEDROCK = os.getenv("USE_BEDROCK", "false").lower() == "true"

if USE_BEDROCK:
    try:
        bedrock = boto3.client("bedrock-runtime")
        logger.info("Initialized AWS Bedrock client")
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {e}")
        raise
else:
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        client = OpenAI(api_key=api_key)
        logger.info("Initialized OpenAI client")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise


class Generator:
    """Class to handle querying an LLM via AWS Bedrock or OpenAI API."""
    
    def __init__(self):
        """Initialize the generator with the appropriate LLM client."""
        self.use_bedrock = USE_BEDROCK
        self.model = (os.getenv("BEDROCK_MODEL", "amazon.titan-text-lite-v1") 
                     if self.use_bedrock 
                     else os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
        logger.info(f"Using model: {self.model}")

    def generate_answer(self, query: str, contexts: List[Dict]) -> str:
        """
        Generate an answer from the given query and retrieved contexts using an LLM.

        Args:
            query (str): The user question.
            contexts (List[Dict]): List of context dictionaries with 'text', 'metadata', and 'score'.

        Returns:
            str: The generated answer or an error message.
        """
        try:
            # Format context from retrieved chunks
            context_str = ""
            for i, ctx in enumerate(contexts, 1):
                text = ctx.get('text', '')
                score = ctx.get('score', 0.0)
                source = ctx.get('metadata', {}).get('source', 'unknown')
                context_str += f"Context {i} (Score: {score:.4f}, Source: {source}):\n{text}\n\n"
            
            if not context_str.strip():
                context_str = "No context provided."

            # Create prompt
            prompt = (
                "You are a helpful assistant. Use the provided context to answer the question concisely and accurately. "
                "If the context is insufficient, say so and provide a general answer if possible.\n\n"
                f"Context:\n{context_str}\n"
                f"Question: {query}\n"
                "Answer:"
            )
            logger.info(f"Generated prompt (first 500 chars): {prompt[:500]}...")

            if self.use_bedrock:
                # AWS Bedrock invocation
                payload = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 512,
                        "temperature": 0.7
                    }
                }
                response = bedrock.invoke_model(
                    modelId=self.model,
                    body=json.dumps(payload),
                    contentType="application/json",
                    accept="application/json"
                )
                response_body = json.loads(response["body"].read())
                answer = response_body.get("results", [{}])[0].get("outputText", "[No output]").strip()
                logger.info("Successfully generated answer with Bedrock")
                return answer

            else:
                # OpenAI API invocation
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512,
                    temperature=0.7
                )
                answer = response.choices[0].message.content.strip()
                logger.info("Successfully generated answer with OpenAI")
                return answer

        except Exception as e:
            logger.error(f"Error during LLM generation: {e}", exc_info=True)
            return f"[LLM Error] {str(e)}"


def main(query: str, context: str):
    """
    Main function for standalone testing.

    Args:
        query (str): The user question.
        context (str): A single context string for testing.
    """
    generator = Generator()
    # Convert single context string to retriever.py-compatible format
    contexts = [{"text": context, "metadata": {"source": "test"}, "score": 1.0}]
    answer = generator.generate_answer(query, contexts)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    """
    Command-line interface to test the generator.

    Example:
        export USE_BEDROCK=false
        export OPENAI_API_KEY=your-key
        python generator.py --query "What is Cloudops tool?" --context "Cloudops is a tool for cloud management..."
    """
    parser = argparse.ArgumentParser(description="Generate answers using an LLM via Bedrock or OpenAI")
    parser.add_argument("--query", default="What is Cloudops tool?", 
                        help="Query string to answer")
    parser.add_argument("--context", default="No context provided", 
                        help="Context string for testing")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set the logging level")

    args = parser.parse_args()

    # Set the logging level
    logger.setLevel(getattr(logging, args.log_level))

    logger.info(f"Starting generation with query: {args.query}")

    try:
        main(args.query, args.context)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)