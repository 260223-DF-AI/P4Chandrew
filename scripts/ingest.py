"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""

import argparse
import os
import uuid
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


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


def load_documents(input_dir: str) -> list:
    """
    Load and return raw documents from the input directory.

    TODO:
    - Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    - Support plain text files.
    - Return a list of Document objects with content and metadata
      (source filename, page number).
    """
    documents = []
    
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path, extract_images=False)
            pages = loader.load()
            
            for page in pages:
                page.metadata['filename'] = filename
                page.metadata['category'] = 'DnD'
            
            documents.extend(pages)
            
    return documents

def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    TODO:
    - Use RecursiveCharacterTextSplitter or sentence-level splitting.
    - Attach chunk metadata (chunk_id, source, page_number, timestamp).
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=150,
        separators=['\n\n','\n', '. ', ' ', '']
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source')}_{i}"
        chunk.metadata["timestamp"] = time.time()
        # source and page_number are usually carried over from the loader
        
    return chunks

def generate_embeddings(chunks: list) -> tuple:
    """
    Generate vector embeddings for document chunks in batches.
    Also prepares the text metadata for ingestion into Pinecone.
    TODO:
    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    embeddings_model = BedrockEmbeddings(
    model_id= 'amazon.titan-embed-text-v2:0',
    region_name= 'us-east-1',
    aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
    model_kwargs={"dimensions": 1024} 
)
    batch_size = 150
    embeddings_list = []
    
    texts = [chunk.page_content for chunk in chunks]
    prepared_data = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        #text = [chunk for chunk in batch]
        batch_embeddings = embeddings_model.embed_documents(batch)

        for j, vector in enumerate(batch_embeddings):
            prepared_data.append({
                    'id': str(uuid.uuid4()),
                    'values': vector,
                    'metadata': {
                        'text': batch[j],
                        'page': chunks[j].metadata['page'],
                        'category': chunks[j].metadata['category'],
                        'timestamp': time.ctime(time.time())
                }
            })

        embeddings_list.extend(batch_embeddings)

        

    return prepared_data
def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    TODO:
    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """
    pinecone = Pinecone(
        api_key= os.getenv('PINECONE_API_KEY')
    )
    
    index_name = os.getenv('PINECONE_INDEX_NAME')
    
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name= index_name,
            dimension= 1024,
            metric= 'cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        
    index = pinecone.Index(name=index_name)
    #ids = [embedding for embedding in embeddings]
    #vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    # upsert_data = []
    # for i, vector in enumerate(embeddings):
    #     upsert_data.append({
    #         "id": str(uuid.uuid4()), # Generate a unique ID
    #         "values": vector
    #     })

    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i : i + batch_size]
        index.upsert(vectors=batch, namespace=namespace)


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()
    print("cli parsed")
    documents = load_documents(args.input_dir)
    print('documents loaded:', len(documents))
    chunks = chunk_documents(documents)
    # for doc_chunk in chunks:
    #     print(f'***{doc_chunk.metadata['page']}***: {doc_chunk.page_content}')
    #     print()
    print('chunks generated:', len(chunks))
    embeddings = generate_embeddings(chunks)
    print('embeddings generated')
    upsert_to_pinecone(embeddings, args.namespace)
    print('embeddings upserted')
    print(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
