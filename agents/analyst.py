"""
ResearchFlow — Analyst Agent

Synthesizes retrieved context into a structured, cited research
response using AWS Bedrock, with Pydantic-validated output.
"""

from pydantic import BaseModel, ValidationError
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
    citations: list[Citation]
    confidence: float  # 0.0 – 1.0


# ---------------------------------------------------------------------------
# Agent Node
# ---------------------------------------------------------------------------

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
    if not chunks:
        # If no chunks exist, we tell the LLM so it returns a low confidence result
        context_str = "ATTENTION: No relevant rule chunks were found in the database for this specific task."
    else:
        formatted_chunks = []
        for i, c in enumerate(chunks, 1):
            formatted_chunks.append(
                f"--- DOCUMENT {i} ---\n"
                f"SOURCE: {c.get('source', 'Unknown')}\n"
                f"PAGE: {c.get('page_number', 'N/A')}\n"
                f"TEXT: {c.get('content', '').strip()}\n"
            )
        context_str = "\n".join(formatted_chunks)
    
    # Build the system and user prompt
    system_prompt = (
    "You are an expert D&D 5e Rules Analyst. You MUST return a valid JSON object.\n"
    "CRITICAL: Do NOT omit any fields. If you have no citations, return an empty list [].\n\n"
    "Example of the EXACT required format:\n"
    "{\n"
    '  "answer": "A grappled creature\'s speed becomes 0.",\n'
    '  "citations": [\n'
    '    {"source": "PHB", "page_number": 290, "excerpt": "A grappled creature\'s speed becomes 0."}\n'
    "  ],\n"
    '  "confidence": 1.0\n'
    "}\n\n"
    "If no data is found, use this structure:\n"
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
    llm = ChatBedrock(
        model_id=os.getenv('BEDROCK_MODEL_ID'),
        model_kwargs={'temperature': 0}
    ).with_structured_output(AnalysisResult)
    
    # Get log for the scratchpad
    initial_logs = state.get("scratchpad", []) + [f"Analyst: Beginning analysis for: {current_subtask}"]
    
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
        "retrieved_chunks": [], 
        "scratchpad": state.get("scratchpad", []) + [log_msg]
    }