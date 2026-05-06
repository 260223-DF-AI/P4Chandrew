"""
Unit Tests — Supervisor Graph

Tests the routing logic and conditional edges using mocked sub-agents.
"""

from unittest.mock import patch, MagicMock
from agents.state import ResearchState
from langgraph.types import interrupt
import agents.supervisor
import pytest


class TestSupervisorRouting:
    """Tests for agents.supervisor routing and conditional edges."""

    def test_planner_decomposes_question(self):
        """
        TODO:
        - Mock the LLM call inside planner_node.
        - Assert it populates state["plan"] with a non-empty list.
        """
        # 1. Create a mock response object where .content is a raw string
        # This string will be split by your '\n' logic
        mock_response = MagicMock()
        mock_response.content = "Retrieve the rule for flanking\nAnalyze combat modifiers\nFact-check 5e SRD"

        # 2. Patch the 'llm' in supervisor.py
        with patch("agents.supervisor.llm") as mock_llm:
            # 3. Setup the mock to return our response object
            mock_llm.invoke.return_value = mock_response

            # 4. Setup state
            state = {
                "question": "How does flanking work?",
                "plan": [],
                "iteration_count": 0,
                "scratchpad": []
            }

            # 5. Call the node
            result = agents.supervisor.planner_node(state)

            # 6. Assertions
            assert "plan" in result
            assert len(result["plan"]) == 3
            assert result["plan"][0] == "Retrieve the rule for flanking"
            assert "PII Masking" in result["scratchpad"][0]
    
    # DONE
    def test_router_selects_retriever(self):
        """
        TODO:
        - Provide a state where the next sub-task requires retrieval.
        - Assert router() returns "retriever".
        """
        from agents.supervisor import router
    
        # Task contains 'retrieve', which matches your 'if any(k...)' logic
        state = {
            "plan": ["Retrieve the rules for grappling"],
            "retrieved_chunks": [],
            "analysis": {}
        }
        
        assert router(state) == "retriever"

    # DONE
    def test_router_selects_analyst(self):
        """
        TODO:
        - Provide a state where retrieval is complete.
        - Assert router() returns "analyst".
        """
        from agents.supervisor import router
        assert router({
            "plan": ["x"],
            "retrieved_chunks": [{"content": "..."}],
        }) == "analyst"

    # DONE
    def test_critique_triggers_retry(self, monkeypatch):
        """
        TODO:
        - Set confidence below threshold, iteration < max.
        - Assert critique_node routes back for refinement.
        """
        monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "3")
        from agents.supervisor import critique_node
        out = critique_node({"confidence_score": 0.3, "iteration_count": 0,
                             "needs_hitl": False})
        assert out["iteration_count"] == 1
        assert out["retrieved_chunks"] == [] 

    def test_critique_triggers_hitl(self, monkeypatch):
        """
        TODO:
        - Set confidence below threshold, iteration >= max.
        - Assert critique_node triggers HITL interrupt.
        """
        monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.6")
        monkeypatch.setenv("MAX_REFINEMENT_ITERATIONS", "2")
        
        # Patch it in the supervisor module before calling the node
        with patch("agents.supervisor.interrupt") as mock_interrupt:
            state = {
                "confidence_score": 0.3, 
                "iteration_count": 2, 
                "needs_hitl": True,
                "scratchpad": []
            }
            
            agents.supervisor.critique_node(state)
            
            mock_interrupt.assert_called_once()

    def test_critique_accepts_response(self, monkeypatch):
        """
        TODO:
        - Set confidence above threshold.
        - Assert critique_node routes to END.
        """
        # 1. Match your env var name and set the threshold
        monkeypatch.setenv("HITL_CONFIDENCE_THRESHOLD", "0.5") 
        
        from agents.supervisor import critique_node

        # 2. Provide the FULL state required by the node logic
        input_state = {
            "confidence_score": 0.8,  # Above 0.5
            "iteration_count": 1,
            "scratchpad": ["Finalizing draft..."],
            "retrieved_chunks": ["chunk1"],
            # Add any other keys the node might try to .get() or access
        }

        # 3. Call the node
        result = critique_node(input_state)

        # 4. Correct Assertions
        # In your code, it returns: "critique": {"status": "Accepted"}
        assert "critique" in result
        assert result["critique"]["status"] == "Accepted"
        
        # Check that it updated the scratchpad correctly
        assert "Critique: Accepted response." in result["scratchpad"][-1]
