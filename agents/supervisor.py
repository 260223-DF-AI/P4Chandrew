"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.
"""

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from dotenv import load_dotenv
from agents.state import ResearchState
from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node
import os
import operator

load_dotenv()
    
llm = ChatBedrock(
        model_id=os.getenv('BEDROCK_MODEL_ID'),
        model_kwargs={'temperature': 0}
    )
    
def planner_node(state: ResearchState) -> dict:
    """
    Decompose the user's question into actionable sub-tasks.

    TODO:
    - Use Bedrock LLM to analyze the question.
    - Return a list of sub-tasks (Plan-and-Execute pattern).
    - Write to the scratchpad for observability.
    """
    
    # bedrock will analyze and plan based on the question
    prompt = f"""You are a D&D 5e assistant. Decompose this question into a simple list of research tasks.
                Each task must start with one of these keywords: 'Retrieve', 'Analyze', or 'Fact-check'.
    Question: {state['question']}
    
    Return each task on a new line."""
    
    response = llm.invoke(prompt)
    
    # Split the response into a list of tasks
    # 'if t.strip()' at the end makes sure that empty strings are not added
    tasks = [t.strip() for t in response.content.split('\n') if t.strip()]
    
    # Manual update for state.py types
    new_scratchpad = state.get("scratchpad", []) + [f"Planner: Identified {len(tasks)} tasks."]
    
    return {
        "plan": tasks,
        "scratchpad": new_scratchpad,
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    
    # Conditional edge: decide which agent to invoke next
    if not state.get("plan"):
        return "fact_checker"
    
    # Check the first task in the list
    current_task = state["plan"][0].lower()
    
    if any(k in current_task for k in ["find", "search", "retrieve", "lookup"]):
        return "retriever"
    return "analyst"
    


def critique_node(state: ResearchState) -> dict:
    """
    Evaluate the aggregated response and decide: accept, retry, or escalate.

    TODO:
    - Check confidence_score against the HITL threshold.
    - If below threshold and iterations < max, loop back for refinement.
    - If below threshold and iterations >= max, trigger HITL interrupt.
    - If above threshold, accept and route to END.
    - Increment iteration_count.
    """
    
    # Evaluate response and decide: accept, retry, or escalate
    MAX_ITERATIONS = 3
    HITL_THRESHOLD = 0.5
    
    score = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)
    current_logs = state.get("scratchpad", [])

    # High Confidence - Finish
    if score >= 0.8:
        return {
            "scratchpad": current_logs + ["Critique: Accepted response."],
            "fact_check_report": {"status": "Accepted"}
        }
    
    # Low Confidence or Max Iterations - Escalate to HITL
    if iterations >= MAX_ITERATIONS or score < HITL_THRESHOLD:
        return {
            "scratchpad": current_logs + ["Critique: Escalating to HITL."],
            "fact_check_report": {"status": "Escalated"}
        }

    # Otherwise - Loop back
    return {
        "scratchpad": current_logs + [f"Critique: Score {score} too low. Retrying."],
        "fact_check_report": {"status": "Retrying"}
    }


def build_supervisor_graph() -> CompiledStateGraph:
    """
    Construct and compile the Supervisor StateGraph.

    TODO:
    - Instantiate StateGraph with ResearchState.
    - Add nodes: planner, retriever, analyst, fact_checker, critique.
    - Add edges and conditional edges (router).
    - Set entry point to planner.
    - Compile and return the graph.

    Returns:
        A compiled LangGraph that can be invoked with an initial state.
    """
    # Construct and compile the Supervisor StateGraph
    builder = StateGraph(ResearchState)

    # Add Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("retriever", retriever_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("critique", critique_node)

    # Define Edges
    builder.add_edge(START, "planner")

    builder.add_conditional_edges(
        "planner",
        router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "fact_checker": "fact_checker"
        }
    )

    # Loop back to planner to refresh the plan/tasks after agent work
    builder.add_edge("retriever", "planner")
    builder.add_edge("analyst", "planner")
    builder.add_edge("fact_checker", "critique")

    # Final routing from critique
    builder.add_conditional_edges(
        "critique",
        lambda state: END if state["fact_check_report"].get("status") in ["Accepted", "Escalated"] else "planner"
    )

    return builder.compile()
 
