"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from pydantic import BaseModel, ValidationError, Field
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from agents.state import ResearchState
from middleware import pii_masking, guardrails
import os

# ---------------------------------------------------------------------------
# Structured Output Schema
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A single supporting citation."""
    source: str
    page_number: int | None = None
    excerpt: str


class AnalysisResult(BaseModel):
    """Pydantic model enforcing structured analyst output."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.0)# 0.0 – 1.0


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

llm = ChatBedrock(
    model_id=os.getenv('BEDROCK_MODEL_ID'),
    model_kwargs={'temperature': 0}
).with_structured_output(AnalysisResult)
    
def analyst_node(state: ResearchState) -> dict:
    """
    Synthesize retrieved chunks into a structured research response.

    TODO:
    - Build a prompt from the question, sub-task, and retrieved_chunks.
    - Invoke AWS Bedrock (e.g., Claude) with structured output enforcement.
    - Parse the response into an AnalysisResult.
    - Support streaming for real-time feedback.
    - Log actions to the scratchpad.

    Returns:
        Dict with "analysis" key containing the AnalysisResult as a dict,
        and "confidence_score" updated from the model's self-assessment.
    """
    
    current_plan = state.get("plan", [])
    current_subtask = current_plan[0] if current_plan else "General Analysis"
    
    # Start building the context by getting the chunks from the retriever
    chunks = state.get("retrieved_chunks", [])
    # print(f"DEBUG: First chunk text: {chunks[0].get('content')[:100] if chunks else 'EMPTY'}")
    # print(f"DEBUG: Analyst is seeing {len(chunks)} chunks.")
    if not chunks:
        # If no chunks exist, we tell the LLM so it returns a low confidence result
        context_str = "ATTENTION: No relevant rule chunks were found in the database for this specific task."
    else:
        formatted_chunks = []
        for i, c in enumerate(chunks, 1):
            formatted_chunks.append(
                f"--- DOCUMENT {i} ---\n"
                f"SOURCE: {c.get('source', 'Unknown')}\n"
                f"PAGE NUMBER: {c.get('page_number', 'N/A')}\n"
                f"TEXT CONTENT: {c.get('content', '').strip()}\n"
            )
        context_str = "\n".join(formatted_chunks)
    
    # Build the system and user prompt
    "TODO: Needs updating.This prompt isn't forcing the LLM to return with all the required keys. (Citations/Confidence)"
    system_prompt = (
        "You are an expert D&D 2024 Rules Analyst. Your role is to synthesize complex "
        "D&D mechanics, abilities, and lore from the provided context into a structured answer.\n\n"
        
        "RULES FOR SYNTHESIS:\n"
        "1. EXHAUSTIVE SCAN: Review ALL provided documents in the 'RETRIEVED RULES CONTEXT'. "
        "The context may contain a mix of specific rules and irrelevant background noise. "
        "Extract every relevant detail that helps answer the user's question.\n"
        "2. NO SELF-AUDITING: Your job is to summarize, NOT verify. Do not discuss if the "
        "information is 'missing' or 'insufficient.' Simply provide the best answer "
        "possible using the available snippets.\n"
        "3. CITATION FIDELITY: Every claim in your 'answer' must be backed by a 'citation'. "
        "The 'excerpt' must be a verbatim (word-for-word) quote from the text.\n"
        "4. 2024 PERSONA: Prioritize 2024 Revised rules terminology. If the context "
        "presents a rule, describe it as it appears in the text.\n\n"
        
        "STRUCTURED OUTPUT REQUIREMENTS:\n"
        "You MUST return a valid JSON object.\n"
        "CRITICAL: Do NOT omit any fields. If you have no citations, return an empty list [].\n"
        "'excerpt' MUST be a verbatim (word-for-word) quote from the 'TEXT CONTENT' used to answer the question.\n\n"
        "Example of the EXACT required format:\n"
        "{\n"
        '  "answer": "A grappled creature\'s speed becomes 0.",\n'
        '  "citations": [\n'
        '    {"source": "PHB", "page_number": 290, "excerpt": "A grappled creature\'s speed becomes 0."}\n'
        "  ],\n"
        '  "confidence": 1.0\n'
        "}\n\n"
        "If no data is found, use this EXACT structure:\n"
        "{\n"
        '  "answer": "No information found.",\n'
        '  "citations": [],\n'
        '  "confidence": 0.0\n'
        "}"
    )
    
    user_content = (
        f"OVERALL QUESTION: {state['question']}\n"
        f"CURRENT STEP: {current_subtask}\n\n"
        f"RETRIEVED RULES CONTEXT:\n{context_str}"
    )
    
    # Apply PII masking and Gaurdrails
    cleaned_prompt, _ = pii_masking.mask_pii(user_content)
    sanitized_prompt = guardrails.sanitize_input(cleaned_prompt)
    
    # Invoke LLM with structured output using the pydantic model
    "TODO: Add Citation Pydantic Model to verify citations are valid."

    # Get log for the scratchpad
    #initial_logs = state.get("scratchpad", []) + [f"Analyst: Beginning analysis for: {current_subtask}"]
    
    try:
        # Attempt the structured call
        result: AnalysisResult = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=sanitized_prompt)
        ])
        
        analysis_data = result.model_dump()
        score = result.confidence
        log_msg = f"Analyst: Success with {len(result.citations)} citations."

    except ValidationError as e:
        # If Pydantic fails, we catch it here so the graph keeps running
        analysis_data = {"answer": "Error: Model failed to produce structured output.", "citations": []}
        score = 0.0
        log_msg = f"Analyst: Pydantic Error - {str(e)}"
        print(f"!!! DEBUG: Pydantic failed: {e}") # This prints to your terminal

    # If validation fails, we must pop the plan to stop an infinite loop
    current_plan = state.get("plan", [])
    new_plan = current_plan[1:] if current_plan else []

    return {
        "analysis": analysis_data,
        "confidence_score": score,
        "plan": new_plan,
        #"retrieved_chunks": [], 
        "scratchpad": [log_msg],
    }