"""
Unit Tests — Analyst Agent

Tests the analyst node using mocked Bedrock calls.
Validates structured output schema and confidence scoring.
"""
import os
os.environ["BEDROCK_MODEL_ID"] = "anthropic.claude-3-5-sonnet-20240620-v1:0"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
from unittest.mock import patch, MagicMock
from agents.analyst import AnalysisResult, Citation, analyst_node
import pytest


class TestAnalystAgent:
    """Tests for agents.analyst.analyst_node."""
    def _stub_result(self):
        return AnalysisResult(
            answer="There are 13 class in DnD 5e.",
            citations=[Citation(source="Player's Handbook.pdf", page_number=12,
                                excerpt="Once the player has chosen from the 13 available classes...")],
            confidence=0.88,
        )
    def test_returns_valid_analysis_result(self):
        """
        TODO:
        - Mock the Bedrock LLM invocation.
        - Call analyst_node with sample retrieved_chunks.
        - Assert the output parses into a valid AnalysisResult.
        """
        # 1. Setup the Mock Result
        mock_res = MagicMock(spec=AnalysisResult)
        mock_res.confidence = 0.9
        mock_res.citations = []
        # This dictionary is what your node actually puts into the state
        mock_dict = {
            "answer": "The rule for grappling is...",
            "citations": [],
            "confidence": 0.9
        }
        mock_res.model_dump.return_value = mock_dict

        # 2. Patch middleware and the global LLM variable
        with patch("agents.analyst.llm") as mock_llm, \
            patch("middleware.pii_masking.mask_pii", return_value=("safe", 0)), \
            patch("middleware.guardrails.sanitize_input", return_value="safe"):
            
            # 3. DIRECT MOCK: The global llm is already the structured one
            mock_llm.invoke.return_value = mock_res

            state = {
                "question": "What is grappling?",
                "plan": ["Analyze grapple"],
                "retrieved_chunks": [],
                "scratchpad": []
            }

            # 4. Execute
            result = analyst_node(state)

            # 5. Assertions
            analysis = result["analysis"]
            
            # Assert it is a dictionary
            assert isinstance(analysis, dict)
            
            # Assert it contains the required keys
            assert "answer" in analysis
            assert "citations" in analysis
            assert "confidence" in analysis
            
            # Bonus: Verify the values match our mock
            assert analysis["confidence"] == 0.9
            assert analysis["answer"] == "The rule for grappling is..."

    def test_includes_citations(self):
        """
        TODO:
        - Assert the AnalysisResult contains at least one Citation.
        - Assert citation source matches a retrieved chunk source.
        """
        from agents.analyst import AnalysisResult, Citation, analyst_node
        
        # 1. Create a real Pydantic object to simulate the structured LLM output
        mock_citation = Citation(source="PHB", page_number=190, excerpt="verbatim text")
        mock_res = AnalysisResult(
            answer="You can grapple by...",
            citations=[mock_citation],
            confidence=1.0
        )

        # 2. Patch the module-level 'llm' object
        with patch("agents.analyst.llm") as mock_structured_llm, \
             patch("middleware.pii_masking.mask_pii", return_value=("safe prompt", 0)), \
             patch("middleware.guardrails.sanitize_input", return_value="safe prompt"):
            
            # Since 'llm' is already the structured output runner, just mock invoke
            mock_structured_llm.invoke.return_value = mock_res

            # 3. Setup state
            state = {
                "question": "How do I grapple?",
                "plan": ["Analyze grappling"],
                "retrieved_chunks": [{"source": "PHB", "content": "Rule text", "page_number": 190}],
                "scratchpad": []
            }

            result = analyst_node(state)

            # 4. Assertions
            analysis = result["analysis"]
            assert len(analysis["citations"]) > 0
            # Fix: Access the index [0] before the key "source"
            assert analysis["citations"][0]["source"] == state["retrieved_chunks"][0]["source"]
            assert result["confidence_score"] == 1.0


    def test_confidence_within_range(self):
        """
        TODO:
        - Assert confidence_score is between 0.0 and 1.0.
        """
        # 1. Create the mock result with the real float
        mock_res = MagicMock(spec=AnalysisResult)
        mock_res.confidence = 0.85
        mock_res.citations = []
        mock_res.model_dump.return_value = {"confidence": 0.85, "citations": [], "answer": "test"}

        # 2. Patch the LLM variable
        with patch("agents.analyst.llm") as mock_llm_var, \
            patch("middleware.pii_masking.mask_pii", return_value=("safe", 0)), \
            patch("middleware.guardrails.sanitize_input", return_value="safe"):
            
            # 3. DIRECT MOCK: Your global llm is already the structured one!
            # Do not mock .with_structured_output() here.
            mock_llm_var.invoke.return_value = mock_res

            state = {
                "question": "test", 
                "plan": ["task"], 
                "retrieved_chunks": [],
                "scratchpad": []
            }
            
            # 4. Execute
            result = analyst_node(state)

            # 5. Assertion
            score = result["confidence_score"]
            
            assert score == 0.85
            assert isinstance(score, float)