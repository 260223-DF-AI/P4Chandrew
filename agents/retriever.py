"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

from agents.state import ResearchState
from langchain_aws import BedrockEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from flashrank import Ranker
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import EmbeddingsFilter
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()


def retriever_node(state: ResearchState) -> dict:
    """
    Retrieve relevant document chunks for the current sub-task.

    TODO:
    - Extract the current sub-task from state["plan"].
    - Query the Pinecone index with semantic search and metadata filters.
    - Apply context compression to reduce token noise.
    - Apply re-ranking to prioritize the most relevant results.
    - Return updated state with retrieved_chunks populated.
    - Log actions to the scratchpad.

    Returns:
        Dict with "retrieved_chunks" key containing a list of dicts,
        each with: content, relevance_score, source, page_number.
    """
    sub_task = state['plan'][0].lower()
    question = state["question"]
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    BEDROCK_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL_ID")

    embeddings_model = BedrockEmbeddings(
    model_id= 'amazon.titan-embed-text-v2:0',
    region_name= os.getenv('AWS_REGION', 'us-east-1'),
    aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
    model_kwargs={"dimensions": 1024}
    )

    embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.01)
    vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), 
                                      embedding=embeddings_model,
                                      namespace='primary-corpus')
    #FlashrankRerank.model_rebuild()
    #compressor = FlashrankRerank(top_n=3)
    compressor = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.5)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
    )
    #print(base_retriever.get_relevant_documents(question))
    docs = compression_retriever.invoke(question)
    for doc in docs:
        print(doc)
    
    
    query_vector = embeddings_model.embed_query(question)
    pc = Pinecone(
        api_key= PINECONE_API_KEY
    )

    index = pc.Index(PINECONE_INDEX_NAME)
    results = index.query(
        vector = query_vector,
        top_k = 5,
        include_metadata = True,
        namespace='primary-corpus'
    )

    #print(results)

if __name__ == "__main__":

    test: ResearchState = {
    'question': "How many classes are there?",
    'plan': [''],
    'retrieved_chunks': [{}],
    'analysis': {},
    'fact_check_report': {},
    'confidence_score': 0.9,
    'iteration_count': 1,
    'scratchpad': list[str],
    'user_id': str
    }

    retriever_node(test)

