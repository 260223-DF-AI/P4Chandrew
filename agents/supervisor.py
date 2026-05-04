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
from middleware import pii_masking
from middleware import guardrails
import os

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
    current_plan = state.get("plan", [])
    analysis = state.get("analysis")
    
    if analysis and not current_plan and state.get("iteration_count", 0) == 0:
        return {
            "scratchpad": state.get("scratchpad", []) + ["Planner: Research cycle 1 complete."]
        }
        
    if not current_plan:
        
        #If an analysis has been made, and we are back in this planner method, we need to make a new plan
        if not analysis:
            # If no analysis has been made, this is our first time in the planning loop
            mode_msg = "Initial Planning"
            prompt = f"""You are a D&D 5e assistant. Decompose this question into a simple list of research tasks.
                    Each task must start with one of these keywords: 'Retrieve', 'Analyze', or 'Fact-check'.
                    Question: {state['question']}
                    Return each task on a new line."""

        else:
            mode_msg = "Plan Refinement"
        
            # bedrock will analyze and plan based on the question
            prompt = f"""The current plan is stuck or needs adjustment. 
                        Original Question: {state['question']}
                        Remaining Tasks: {current_plan}
                        Recent Activity: {state.get('scratchpad', [])[-2:]}
                        Provide an updated list of tasks."""
        
    # else:
    #     # If we are here and a plan exists, we just want to log that we are continuing
    #     return {
    #         "plan": current_plan,
    #         "scratchpad": state.get("scratchpad", []) + ["Planner: Continuing with existing plan."]
    #     }

    
        cleaned_prompt, prompt_redactions = pii_masking.mask_pii(prompt)
        sanitized_cleaned_prompt = guardrails.sanitize_input(cleaned_prompt)
        response = llm.invoke(sanitized_cleaned_prompt)
        cleaned_response, response_redactions = pii_masking.mask_pii(response.content)
        
        total_redactions = prompt_redactions + response_redactions
        # Make a log for the scratchpad with the number of redactions
        pii_log = f"PII Masking ({mode_msg})\nTotal Redactions: {total_redactions}\nPrompt Redactions: {prompt_redactions}\nResponse Redactions: {response_redactions}"

        # Split the response into a list of tasks
        # 'if t.strip()' at the end makes sure that empty strings are not added
        tasks = [t.strip() for t in cleaned_response.split('\n') if t.strip()]
        
        # Combine log with the task list
        new_logs = [pii_log, f"Planner: {mode_msg} Identified {len(tasks)} tasks."]
        
        return {
            "plan": tasks,
            "scratchpad": state.get("scratchpad", []) + new_logs,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    return {"plan": state.get("plan", []), "scratchpad": state.get("scratchpad", []) + ["Planner: Proceeding..."]}


def router(state: ResearchState) -> str:
    """
    Conditional edge: decide which agent to invoke next.

    TODO:
    - Inspect the current plan and state to choose the next node.
    - Return the node name as a string (used by add_conditional_edges).
    """
    
    plan = state.get("plan", [])
    chunks = state.get("retrieved_chunks", [])
    analysis = state.get("analysis")
    fact_checked = state.get("fact_check_report")
    
    # If there's a task, decide based on whether we have data yet
    if plan:
        current_task = plan[0].lower()
        is_retrieval_task = any(k in current_task for k in ["find", "search", "retrieve", "lookup"])
        
        # If it's a retrieval task and we don't have results yet, go get them
        if not chunks:
            return "retriever"
        
        # Otherwise, the Analyst handles the task (and pops it when done)
        return "analyst"

    # Plan is empty - Check for finalization steps
    if analysis:
        # If we have an answer but haven't fact-checked yet, go there
        if not fact_checked:
            return "fact_checker"
        
        # Both analysis and fact-check are done, go to critique to finish
        return "critique"

    # Default fallback if somehow we have no plan and no analysis
    return "planner"


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
    HITL_THRESHOLD = 0.7
    
    score = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)
    current_logs = state.get("scratchpad", [])

    # High Confidence - Finish
    if score >= HITL_THRESHOLD:
        return {
            "scratchpad": current_logs + ["Critique: Accepted response."],
            "fact_check_report": {"status": "Accepted"}
        }
    
    # Low Confidence but we have retries left - Loop back
    if score < HITL_THRESHOLD and iterations < MAX_ITERATIONS:
        return {
            "iteration_count": iterations + 1, # Explicitly increment state
            "scratchpad": current_logs + [f"Critique: Score {score} too low. Retrying ({iterations + 1}/{MAX_ITERATIONS})."],
            "fact_check_report": {"status": "Retrying"}
        }

    # Max Iterations reached or critically low score - Escalate
    return {
        "scratchpad": current_logs + ["Critique: Max iterations reached or threshold failed. Escalating to HITL."],
        "fact_check_report": {"status": "Escalated"}
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

    # Loop back to planner to refresh the plan/tasks after agent work, as its good practice
    # builder.add_edge("retriever", "planner")
    
    # Go directly to analyst, skip looping back to planner as it's not needed for our use case, and wastes tokens
    builder.add_edge("retriever", "analyst")
    # Fact checker goes to critique to very the quality of the response
    builder.add_edge("fact_checker", "critique")
    builder.add_conditional_edges(
        "analyst",
        router,
        {
            "retriever": "retriever",
            "analyst": "analyst",
            "fact_checker": "fact_checker",
            "critique": "critique", # <--- This is the exit ramp!
            "planner": "planner"
        }
    )
    # builder.add_edge("fact_checker", "critique")

    # Final routing from critique
    builder.add_conditional_edges(
        "critique",
        lambda state: END if state["fact_check_report"].get("status") in ["Accepted", "Escalated"] else "planner"
    )

    return builder.compile()
 
if __name__ == "__main__":
    # Compile the graph
    app = build_supervisor_graph()

    # Setup a realistic D&D query
    # "How does the 'Grappled' condition affect movement?" is a great test case
    inputs = {
        "question": "How does the Grappled condition affect movement?",
        "plan": [],
        "retrieved_chunks": [],
        "analysis": {},
        "fact_check_report": {},
        "confidence_score": 0.0,
        "iteration_count": 0,
        "scratchpad": [],
        "user_id": "local_dev_123"
    }

    print("\n=== STARTING RESEARCH FLOW ===")
    
    # Stream the events so you can see the nodes working in real-time
    for event in app.stream(inputs, stream_mode="updates"):
        for node_name, output in event.items():
            print(f"\n--- Output from Node: {node_name} ---")
            
            # Print the current plan to see if it's being "popped" correctly
            if "plan" in output:
                print(f"Current Plan: {output['plan']}")
            
            # Print the number of chunks retrieved
            if "retrieved_chunks" in output:
                print(f"Retrieved {len(output['retrieved_chunks'])} chunks.")

            # Print the synthesis if the Analyst finished
            if "analysis" in output:
                print(f"Analysis Answer: {output['analysis'].get('answer')[:200]}...")
                
            if node_name == "critique":
                report = output.get("fact_check_report", {})
                print(f"Status: {report.get('status')}")
                # Ensure your critique node writes to the scratchpad so you see it here
                for log in output.get("scratchpad", []):
                    print(f"Log: {log}")

    print("\n=== FLOW COMPLETE ===")
