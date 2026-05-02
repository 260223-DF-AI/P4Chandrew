"""
ResearchFlow — Retriever Agent

Queries the Pinecone vector store using semantic search,
applies context compression and re-ranking, and returns
structured retrieval results to the Supervisor.
"""

# Cohere requires langchain_classic, 'mock' it by passing in the normal langchain as 'langchain_classic'
# This should be before the langchain_cohere import
import sys
try:
    import langchain
    sys.modules["langchain_classic"] = langchain
except ImportError:
    # If for some reason 'langchain' isn't found, 
    # we just bypass the error so the script can continue
    pass

from agents.state import ResearchState
from langchain_aws import BedrockEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_pinecone import PineconeVectorStore
from langchain.retrievers import ContextualCompressionRetriever
#from flashrank import Ranker
from langchain_cohere import CohereRerank
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers.document_compressors import EmbeddingsFilter
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

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

# Testing Flashrank to rerank
#FlashrankRerank.model_rebuild()
#compressor = FlashrankRerank(top_n=3)

# Testing EmbeddingsFilter to rerank
#compressor = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.5)

# Testing Cohere to rerank
compressor = CohereRerank(model="rerank-english-v3.0")

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compression_retriever = ContextualCompressionRetriever(
base_compressor=compressor, 
base_retriever=base_retriever
)

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
    # If planner hasn't run yet, use the question as the subtask
    current_plan = state.get("plan", [])
    current_subtask = current_plan[0] if current_plan else state["question"]
    
    docs = compression_retriever.invoke(current_subtask)

    formatted_chunks = []
    for doc in docs:
        formatted_chunks.append({
            "content": doc.page_content, # 'text' is automatically mapped here by PineconeVectorStore
            "relevance_score": doc.metadata.get("relevance_score", "N/A"),
            "category": doc.metadata.get("category", "Unknown"), # Replaces 'source'
            "page_number": doc.metadata.get("page", "N/A"),
            "timestamp": doc.metadata.get("timestamp", "N/A")
        })
        
    log_entry = f"Retriever Agent: Searched for '{current_subtask}' and found {len(formatted_chunks)} relevant chunks."
    
    # Pop off the current task, so the planner can continue with the next task
    remaining_plan = state.get("plan", [])
    if remaining_plan:
        remaining_plan.pop(0) 
        
    # Return the keys that need updating in the shared state
    return {
        "plan": remaining_plan,
        "retrieved_chunks": formatted_chunks,
        "scratchpad": state.get("scratchpad", []) + [log_entry]
    }

if __name__ == "__main__":

    test_state: ResearchState = {
        'question': "What is the best class in Dungeons & Dragons for level 1?",
        'plan': ["List all classes in Dungeons & Dragons."],
        'retrieved_chunks': [],
        'analysis': {},
        'fact_check_report': {},
        'confidence_score': 0.0,
        'iteration_count': 0,
        'scratchpad': [],
        'user_id': "local_dev_user"
    }
    
    print(f"--- Executing Retriever Node Test ---")
    result = retriever_node(test_state)
    
    for i, chunk in enumerate(result["retrieved_chunks"]):
        print(f"\n[Rank {i+1}] Score: {chunk['relevance_score'] * 100:.2f}%")
        print(f"Category: {chunk['category']} | Page: {chunk['page_number']}")
        print(f"Snippet: {chunk['content'][:600]}...")

