"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/primary-corpus --namespace primary-corpus
    python scripts/ingest.py --input-dir ./data/fact-check-sources --namespace fact-check-sources
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_aws import BedrockEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from datetime import datetime
import argparse
import os
import uuid
import time
import re



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

    # Initialize Markdown text splitter
    headers_to_split_on = [
        ("#", "H1"),   # Captures whatever follows # as metadata['H1']
        ("##", "H2"),  # Captures whatever follows ## as metadata['H2']
        ("###", "H3"),  # Captures whatever follows ### as metadata['H3']
        ("####", "H4"),  # Captures whatever follows #### as metadata['H4']
        ("#####", "H5")  # Captures whatever follows ##### as metadata['H5']
    ]
    
    # Iterate over files in the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            
            ctime = os.path.getctime(file_path)
            creation_date = datetime.fromtimestamp(ctime).strftime('%Y-%m-%d')
            
            # --- RESEARCH DATA (Markdowns) ---
            if filename.endswith('.md') and "README" not in filename:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split and add folder hierarchy metadata
                md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_chunks = md_splitter.split_text(content)

                rel_dir = os.path.relpath(root, input_dir)
                for chunk in md_chunks:
                    chunk.metadata.update({
                        'filename': filename,
                        'subject': rel_dir if rel_dir != "." else "general",
                        'category': 'DnD_Research',
                        'source': file_path,
                        'creationdate': creation_date,
                        'timestamp': str(datetime.fromtimestamp(time.time()))
                    })
                documents.extend(md_chunks)
                
            # --- FACT-CHECK DATA (PDFs) ---
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path, extract_images=False)
                pages = loader.load()
                
                for page in pages:
                    page.metadata.update({
                        'filename': filename,
                        'category': 'DnD_Official_Rules',
                        'subject': 'Official',
                        'source': file_path,
                        'creationdate': creation_date,
                        'timestamp': str(datetime.fromtimestamp(time.time()))
                        
                    })
                documents.extend(pages)

    return documents

def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    TODO:
    - Use RecursiveCharacterTextSplitter or sentence-level splitting.
    - Attach chunk metadata (chunk_id, source, page_number, timestamp).
    """
    final_chunks = []
    
    pdf_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200,
        separators=['\n\n', '\n', '. ', ' ', '']
    )

    md_safety_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=300,
        separators=['\n\n', '\n', '. ', ' ', '']
    )
    
    for doc in documents:
    # This replaces single \n with a space, but keeps \n\n (paragraphs) intact
    # We do this in case the page changes mid-paragraph and we want to preserve the paragraph
        doc.page_content = re.sub(r'(?<!\n)\n(?!\n)', ' ', doc.page_content)
        
        is_markdown = "H1" in doc.metadata or "H2" in doc.metadata

        if is_markdown:
            # If the Markdown section is within a safe size, keep it whole.
            # If it's massive (over 4k chars), force-split it to avoid AWS 400 errors.
            if len(doc.page_content) > 4000:
                sub_docs = md_safety_splitter.split_documents([doc])
                final_chunks.extend(sub_docs)
            else:
                final_chunks.append(doc)
        else:
            # It's a PDF page. ALWAYS split it for better retrieval granularity.
            pdf_chunks = pdf_splitter.split_documents([doc])
            final_chunks.extend(pdf_chunks)
        
    # for i, chunk in enumerate(final_chunks):
    #     chunk.metadata["id"] = f"{chunk.metadata.get('source')}_{i}"
    #     chunk.metadata["timestamp"] = time.time()
        # source and page_number are usually carried over from the loader
        
    return final_chunks

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
    region_name= os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
    model_kwargs={"dimensions": 1024} 
    )

    batch_size = 50
    #embeddings_list = []
    prepared_data = []
    texts = [chunk.page_content for chunk in chunks]

    # loop through the text in chunks, embedding them in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_chunks = chunks[i : i + batch_size]
        
        batch_embeddings = embeddings_model.embed_documents(batch_texts)

        # set the metadata for each embedding
        for j, vector in enumerate(batch_embeddings):
            chunk_text = batch_texts[j]
            
            # Inside many markdown files there is a source. Find it and remove any styling from it
            source_pattern = r"(?:Source|source):\s*[\*_]*([^\*_\n\r]+)[\*_]*"
            match = re.search(source_pattern, chunk_text)
            
            # If found, use it. else default to the subject (like 'diseases')
            internal_citation = match.group(1).strip() if match else batch_chunks[j].metadata.get('subject', 'General')
            prepared_data.append({
                    'id': str(uuid.uuid4()),
                    'values': vector,
                    'metadata': {
                        'filename': batch_chunks[j].metadata.get('filename', 'unknown'),
                        'text': chunk_text,
                        'cited_source': internal_citation,
                        'source': batch_chunks[j].metadata.get('source', 'unknown'),
                        'page': batch_chunks[j].metadata.get('page', 0),
                        'category': batch_chunks[j].metadata.get('category', 'DnD'),
                        'subject': batch_chunks[j].metadata.get('subject', 'general'),
                        'creationdate': batch_chunks[j].metadata.get('creationdate', 'unknown'),
                        'timestamp': batch_chunks[j].metadata.get('timestamp', 0),
                        'H1': batch_chunks[j].metadata.get('H1', ''),
                        'H2': batch_chunks[j].metadata.get('H2', ''),
                        'H3': batch_chunks[j].metadata.get('H3', ''),
                        'H4': batch_chunks[j].metadata.get('H4', ''),
                        'H5': batch_chunks[j].metadata.get('H5', '')
                }
            })

        #embeddings_list.extend(batch_embeddings)
        print(f"\t- Generated {i + len(batch_texts)} / {len(texts)} embeddings...")
    # return prepared_data dictionary, which has 'values' as the key for embeddings
    return prepared_data
def upsert_to_pinecone(embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    TODO:
    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """

    # Get embeddings set up
    embeddings_model = BedrockEmbeddings(
    model_id= 'amazon.titan-embed-text-v2:0',
    region_name= os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
    model_kwargs={"dimensions": 1024} 
    )

    # Create Pinecone object with API key
    pinecone = Pinecone(
        api_key= os.getenv('PINECONE_API_KEY')
    )
    
    # Establish index
    index_name = os.getenv('PINECONE_INDEX_NAME')
    
    # If index doesn't exist, create it
    # Make sure the embedding model is set up to allow text search in Pinecone dashboard (embed with llama-text-embed-v2)
#     if not pinecone.has_index(index_name):
#         pinecone.create_index_for_model(
#         name=index_name,
#         cloud="aws",
#         region=os.getenv('AWS_REGION', 'us-east-1'),
#         # This is another way to use serverless spec for embeddings
#         embed={
#             "model": "llama-text-embed-v2",
#             "field_map": {"text": "text"} # "Look in metadata['text'] for the searchable content"
#         }
# )

    # Use for Hybrid search
    if not pinecone.has_index(index_name):
        pinecone.create_index(
            name=index_name,
            dimension=1024,
            metric="dotproduct", # Required for Hybrid search
            spec=ServerlessSpec(
                cloud="aws",
                region=os.getenv('AWS_REGION', 'us-east-1')
            )
        )
    # Initialize the vectorstore
    vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), 
                                      embedding=embeddings_model,
                                      namespace=namespace)
    
    index = pinecone.Index(name=index_name)

    batch_size = 100
    print(f"\t- Upserting to namespace: {namespace}...")
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i : i + batch_size]
        
        docs = []
        for item in batch:
            # 1. Extract the text from the nested metadata
            # Adjust the keys if your structure is slightly different
            metadata_dict = item.get("metadata", {})
            page_content = metadata_dict.get("text", "")
            
            # 2. Create the Document
            # We pass the whole 'item' or just the 'metadata' dict as metadata
            docs.append(Document(page_content=page_content, metadata=metadata_dict))
        # 3. Add the batch to Pinecone via LangChain
        vectorstore.add_documents(docs, namespace=namespace)
        print(f"\t- Upserted {i + len(batch)} / {len(embeddings)} embeddings...")

def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()
    print("cli parsed")
    documents = load_documents(args.input_dir)
    print('documents loaded:', len(documents))
    chunks = chunk_documents(documents)
    print('chunks generated:', len(chunks))
    embeddings = generate_embeddings(chunks)
    print('embeddings generated')
    upsert_to_pinecone(embeddings, args.namespace)
    print('embeddings upserted')
    print(f"Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
