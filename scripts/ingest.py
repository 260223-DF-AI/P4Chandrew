"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""

import argparse
import os

from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document


def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing PDF/text documents.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="primary-corpus",
        help="Pinecone namespace to upsert into.",
    )
    return parser.parse_args()

def _pdf_reader(file_path: str):
    """Load a PDF file yielding content of a page w/ page number attached."""
    reader = PdfReader(file_path)

    for page_num, page in enumerate(reader.pages):
        extract = page.extract_text()

        text = [line for line in extract.split("\n") if len(line.strip()) > 0] 
        yield (page_num, "\n".join(text))

def load_documents(input_dir: str) -> list:
    """
    Load and return raw documents from the input directory.

    TODO:
    x Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    x Support plain text files.
    x Return a list of Document objects with content and metadata
      (source filename, page number).
    """
    documents : list[Document] = []

    # returned source is just the file path and not just the file name
    if input_dir.endswith(".pdf"):
        for page_num, text in _pdf_reader(input_dir):
            content = Document(page_content=text, metadata={"source": input_dir, "page": page_num})
            documents.append(content)
    
    elif input_dir.endswith(".txt"):
        with open(input_dir, "r") as f:
            text = f.read()
            content = Document(page_content=text, metadata={"source": input_dir, "page": 0})
            documents.append(content)

    # temp. for now unless we only want to support .pdf and .txt
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf or .txt file.")
    
    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    TODO:
    - Use RecursiveCharacterTextSplitter or sentence-level splitting.
    - Attach chunk metadata (chunk_id, source, page_number, timestamp).
    """
    raise NotImplementedError


def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    TODO:
    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    raise NotImplementedError


def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    TODO:
    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """
    raise NotImplementedError


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    upsert_to_pinecone(embeddings, args.namespace)

    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
