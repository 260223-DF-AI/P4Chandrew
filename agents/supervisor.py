"""
ResearchFlow — Supervisor Graph

Builds and returns the main LangGraph StateGraph that orchestrates
the Planner, Retriever, Analyst, Fact-Checker, and Critique nodes.

Usage:
    python agents/supervisor.py --query "Your question here"
    
example:
    python agents/supervisor.py --query "How does the Grappled condition affect movement?"
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
from langgraph.types import interrupt
import os
import argparse

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
            "scratchpad": ["Planner: Research cycle 1 complete."]
        }
        
    if not current_plan:
        
        if not analysis:
            # If no analysis has been made, this is our first time in the planning loop
            mode_msg = "Initial Planning"
            prompt = f"""You are a D&D assistant. Decompose this question into a simple list of research tasks.
                    Each task must start with one of these keywords: 'Retrieve', 'Analyze', or 'Fact-check'.
                    Question: {state['question']}
                    Return each task on a new line."""

        #If an analysis has been made, and we are back in this planner method, we need to make a new plan
        else:
            mode_msg = "Plan Refinement"
        
            # bedrock will analyze and plan based on the question
            prompt = f"""The current plan is stuck or needs adjustment. 
                        Original Question: {state['question']}
                        Remaining Tasks: {current_plan}
                        Recent Activity: {state.get('scratchpad', [])[-2:]}
                        You are a D&D assistant. Decompose the original questions into an updated list of tasks. 
                        Return each task on a new line, without additonal formatting. 
                        Do not add any numbers for listing the tasks, bullet points, hyphens, quotes, nor asterisks.
                        """

        # Clean the user prompt and LLM response
        cleaned_prompt, prompt_redactions = pii_masking.mask_pii(prompt)
        sanitized_cleaned_prompt = guardrails.sanitize_input(cleaned_prompt)
        response = llm.invoke(sanitized_cleaned_prompt)
        cleaned_response, response_redactions = pii_masking.mask_pii(response.content)
        

        # Make a log for the scratchpad with the number of redactions
        total_redactions = prompt_redactions + response_redactions
        pii_log = f"PII Masking ({mode_msg})\nTotal Redactions: {total_redactions}\nPrompt Redactions: {prompt_redactions}\nResponse Redactions: {response_redactions}"

        # Split the response into a list of tasks. These will go to the other nodes for proceessing
        tasks = [t.strip() for t in cleaned_response.split('\n') if t.strip()]
        
        # Combine log with the task list
        new_logs = [pii_log, f"Planner: {mode_msg} Identified {len(tasks)} tasks."]
        
        return {
            "plan": tasks,
            "scratchpad": new_logs,
            "iteration_count": state.get("iteration_count", 0) + 1
        }
    
    # If there is a plan and analysis, return current plan and keep going
    return {"plan": state.get("plan", []), "scratchpad": ["Planner: Proceeding..."]}


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
    
    # if analysis and not fact_checked:
    #     return "fact_checker"

    # # If there's a task, decide based on whether we have data yet
    # if plan:
    #     current_task = plan[0].lower()
    #     #is_retrieval_task = any(k in current_task for k in ["find", "search", "retrieve", "lookup"])
        
    #     # If it's a retrieval task and we don't have results yet, go get them
    #     if not chunks:
    #         return "retriever"
        
    #     # Otherwise, the Analyst handles the task (and pops it when done)
    #     return "analyst"

    # # Plan is empty - Check for finalization steps
    # if analysis:
    #     # If we have an answer but haven't fact-checked yet, go there
    #     if not fact_checked:
    #         return "fact_checker"
        
    #     # Both analysis and fact-check are done, go to critique to finish
    #     return "critique"

    # # Default fallback if somehow we have no plan and no analysis
    # return "planner"

    plan = state.get("plan", [])
    
    # 1. Check for finished plan first!
    if not plan:
        # Once tasks are done, check if we need to fact-check the final result
        if state.get("analysis") and not state.get("fact_check_report"):
            return "fact_checker"
        return "critique" # Final node

    # 2. Map current task to a node
    current_task = plan[0].lower()
    
    if any(k.lower() in current_task for k in ["find", "search", "retrieve"]):
        return "retriever"
    
    if any(k.lower() in current_task for k in ["analyze", "summarize", "write"]):
        return "analyst"
    
    if any(k.lower() in current_task for k in ["fact-check", "verify", "check"]):
        return "fact_checker"
        
    return "analyst" # Default fallback

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
    MAX_ITERATIONS = int(os.getenv("MAX_REFINEMENT_ITERATIONS", 3))
    HITL_THRESHOLD = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", 0.6))
    
    score = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)
    current_logs = state.get("scratchpad", [])

    #TODO: Make sure the Critique node writes to the newly added critique field in state, instead of scratchpad
    #TODO: Check if adding retrieves chunks to state is necessary for testing, if not remove
    # High Confidence - Finish
    if score >= HITL_THRESHOLD:
        return {
            "scratchpad": current_logs + ["Critique: Accepted response."],
            "critique": {"status": "Accepted"},
            "retrieved_chunks": state.get("retrieved_chunks", [])
        }
    
    # Low Confidence but we have retries left - Loop back
    if score < HITL_THRESHOLD and iterations < MAX_ITERATIONS:
        return {
            "iteration_count": iterations + 1, # Explicitly increment state
            "scratchpad": current_logs + [f"Critique: Score {score} too low. Retrying ({iterations + 1}/{MAX_ITERATIONS})."],
            "critique": {"status": "Retrying"},
            "retrieved_chunks": state.get("retrieved_chunks", [])
        }

    # Max Iterations reached or critically low score - Escalate
    escalated_state = {
        "scratchpad": current_logs + ["Critique: Max iterations reached or threshold failed. Escalating to HITL."],
        "critique": {"status": "Escalated"},
        "retrieved_chunks": state.get("retrieved_chunks", [])
    }
    return interrupt(escalated_state)



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
            "critique": "critique",
            "planner": "planner"
        }
    )
    # builder.add_edge("fact_checker", "critique")

    # Final routing from critique
    builder.add_conditional_edges(
        "critique",
        lambda state: END if state["critique"].get("status") in ["Accepted", "Escalated"] else "planner"
    )

    return builder.compile()

def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    parser = argparse.ArgumentParser(description="Query to pass into ResearchFlow.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="the query that the supervisor will process",
    )
    return parser.parse_args()
 
if __name__ == "__main__":
    # Compile the graph
    app = build_supervisor_graph()

    args = parse_args()
    query = args.query if args.query else "How does the Grappled condition affect movement?"
    # Setup a realistic D&D query
    # "How does the 'Grappled' condition affect movement?" is a great test case
    inputs = {
        "question": query,
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
    # Program is crashing due to recursion limit. Change from 25 to 50
    config={"recursion_limit": 50} 

    # Stream the events so you can see the nodes working in real-time
    for event in app.stream(inputs, stream_mode="updates", config=config):
        for node_name, output in event.items():
            print(f"\n--- Output from Node: {node_name} ---")
            
            # Print the current plan to see if it's being "popped" correctly
            if "plan" in output:
                print(f"Current Plan: {output['plan']}")
            
            # Print the number of chunks retrieved
            if "retrieved_chunks" in output and node_name == "retriever":
                print(f"Retrieved {len(output['retrieved_chunks'])} chunks.")

            # Print the synthesis if the Analyst finished
            if "analysis" in output:
                print(f"Analysis Answer: {output['analysis'].get('answer')[:500]}...")
            
            if "fact_check_report" in output:
                print(f"Fact Check Status: {output['fact_check_report'].get('status')}")

            if "critique" in output:
                print(f"Critique Status: {output['critique'].get('status')}")



            if node_name == "critique":
                report = output.get("critique", {})
                print(f"Status: {report.get('status')}")
            # Ensure your critique node writes to the scratchpad so you see it here
            
            if "scratchpad" in output:
                scratchpad = output.get("scratchpad")
                if scratchpad:
                    for log in scratchpad:
                        print(f"Log: {log}")

    print("\n=== FLOW COMPLETE ===")