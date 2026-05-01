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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
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
    - Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    - Support plain text files.
    - Return a list of Document objects with content and metadata
      (source filename, page number).
    """
    documents: list[Document] = []

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


def _add_document_metadata(doc, new_metadata):
    old_metadata = doc.metadata
    new_metadata = {**old_metadata, **new_metadata}
    return Document(page_content=doc.page_content, metadata=new_metadata)


def chunk_documents(documents: list[Document]) -> list:
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
    chunks: list[dict] = []

    for document in documents:
        split_text = splitter.split_documents([document])
        for num, chunk in enumerate(split_text):
            chunks.append(
                {
                    "_id": str(chunk_id),
                    "chunk_text": chunk.page_content,
                    **_add_document_metadata(chunk, {"chunk_num": num}).metadata,
                }
            )
            chunk_id += 1

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

    # loop through the text in chunks, embedding them in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embeddings_model.embed_documents(batch)

        # set the metadata for each embedding
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

    # return prepared_data dictionary, which has 'values' as the key for embeddings
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