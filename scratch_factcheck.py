import os
from dotenv import load_dotenv
load_dotenv()

from agents.retriever import retriever_node
from agents.analyst import analyst_node
from agents.fact_checker import fact_checker_node

state = {
    "question": "How many classes are there in Dungeons & Dragons?",
    "plan": [],
    "current_subtask_index": 0,
}
state.update(retriever_node(state))
state.update(analyst_node(state))
state.update(fact_checker_node(state))

print("Overall confidence:", state["confidence_score"])
print("Needs HITL:", state["needs_hitl"])
for v in state["fact_check_report"]["verdicts"]:
    print(f"  [{v['verdict']}] {v['claim'][:60]}...")