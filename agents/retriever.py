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
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
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

encoder = BM25Encoder().default()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index_instance = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

#embeddings_filter = EmbeddingsFilter(embeddings=embeddings_model, similarity_threshold=0.01)
vectorstore = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), 
                                    embedding=embeddings_model,
                                     text_key="text",
                                    namespace='primary-corpus')

#base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
base_retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings_model, 
    sparse_encoder=encoder,
    index=pinecone_index_instance, # The raw Pinecone Index object
    top_k=50,
    alpha=0.7, # 0.7 focuses on keywords, 0.3 on semantic
    text_key="text",
    namespace="primary-corpus"
)

# Use Cohere to rerank
compressor = CohereRerank(model="rerank-english-v3.0", top_n=20)
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
    # Get the plan list from state. If it doesn't exist, use an empty list
    current_plan = state.get("plan", [])
    # If planner hasn't run yet, use the question as the subtask
    current_subtask = current_plan[0] if current_plan else state["question"]
    # If the current subtask is a fact check, use the fact check corpus
    if current_subtask.lower().startswith("fact-check"):
        target_ns = "fact-check-sources"
    else:
        target_ns = "primary-corpus"
        
    base_retriever.namespace = target_ns
    current_subtask = current_subtask.replace("Retrieve", "").replace("Analyze", "").replace("Fact-check", "").strip()
    print(f"\n---Query: {current_subtask}---\n")
    
    # Define the rule folders and rule files, since other files like adventures are giving poor results
    rule_folders = ["books", "variant-rules"]
    rule_files = ["conditions.md", "actions.md"]

    # Build the filter: 
    # Must NOT be in adventures AND (must be in rule_folders OR be a rule_file)
    search_filter = {
        "$and": [
            {"subject": {"$not": {"$regex": ".*adventures.*"}}}, 
            {"$or": [
                {"subject": {"$regex": f".*({'|'.join(rule_folders)}).*"}},
                {"subject": {"$in": rule_files}}
            ]}
        ]
    }
    
    # Query the Pinecone index, apply compression, and re-rank with fresh configurations targeting the correct namespace
    docs = compression_retriever.invoke(current_subtask,
                                        config={"configurable": {
                                            "search_kwargs": {
                                                "filter": search_filter, 
                                                #"namespace": target_ns
                                                }
                                            }
                                        })
    
    print(f"Retrieved {len(docs)} chunks after reranking.")
    formatted_chunks = []
    for doc in docs:
        formatted_chunks.append({
            #"text": doc.page_content, # 'text' is automatically mapped here by PineconeVectorStore
            "content": doc.metadata.get("text", doc.page_content),
            "relevance_score": doc.metadata.get("relevance_score", "N/A"),
            "category": doc.metadata.get("category", "Unknown"), # Replaces 'source'
            "page_number": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "N/A"),
            "timestamp": doc.metadata.get("timestamp", "N/A")
        })
        
    log_entry = f"Retriever Agent: Searched for '{current_subtask}' in {target_ns} and found {len(formatted_chunks)} chunks after reranking."
    
    # Return the keys that need updating in the shared state
    return {
        #"plan": remaining_plan,
        "retrieved_chunks": formatted_chunks,
        "scratchpad": [log_entry]
    }

if __name__ == "__main__":

    test_state: ResearchState = {
        'question': "How many classes are there in Dungeons & Dragons.",
        'plan': [],
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
        print(f"Source: {chunk['source']}")
        print(f"Category: {chunk['category']} | Page: {chunk['page_number']}")
        print(f"Snippet: {chunk['content'][:600]}...")

