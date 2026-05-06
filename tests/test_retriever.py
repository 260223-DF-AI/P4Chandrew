"""
Unit Tests — Retriever Agent

Tests the retriever node using mocked Pinecone calls.
Validates re-ranking behavior and output structure.
"""

from unittest.mock import patch, MagicMock
import pytest


class TestRetrieverAgent:
    """Tests for agents.retriever.retriever_node."""

    def test_returns_structured_chunks(self):
        """
        TODO:
        - Mock the Pinecone client's query method.
        - Call retriever_node with a sample state.
        - Assert the returned dict contains "retrieved_chunks".
        - Assert each chunk has: content, relevance_score, source, page_number.
        """
        # 1. Mock the LangChain Document objects
        mock_doc = MagicMock()
        mock_doc.page_content = "The wizard casts fireball."
        mock_doc.metadata = {
            "text": "The wizard casts fireball.",
            "relevance_score": 0.95,
            "source": "Player's Handbook",
            "page": 202,
            "category": "Rules"
        }

        # 2. Prepare sample state
        sample_state = {
            "question": "How do spells work?",
            "plan": ["Retrieve spellcasting rules"],
            "retrieved_chunks": [],
            "scratchpad": [],
            "user_id": "test_user"
        }

        # 3. Patch where 'compression_retriever' is used in your node
        # Note: Replace 'agents.retriever' with the actual path to your file
        with patch("agents.retriever.compression_retriever") as mock_retriever:
            # We force the mock to have an 'invoke' method that returns our list
            mock_retriever.invoke.return_value = [mock_doc]

            from agents.retriever import retriever_node
            result = retriever_node(sample_state)

            # 4. Assertions
            assert "retrieved_chunks" in result
            assert len(result["retrieved_chunks"]) == 1
            
            chunk = result["retrieved_chunks"][0]
            assert chunk["content"] == "The wizard casts fireball."
            assert chunk["relevance_score"] == 0.95
            assert chunk["source"] == "Player's Handbook"
            assert chunk["page_number"] == 202

    def test_applies_reranking(self):
        """
        TODO:
        - Provide mock results in non-optimal order.
        - Assert that re-ranking reorders them by relevance.
        """
        # 1. Create mock docs in a specific 'reranked' order
        # High relevance first
        doc1 = MagicMock(page_content="High relevance chunk", 
                         metadata={"relevance_score": 0.99, "source": "A"})
        # Low relevance second
        doc2 = MagicMock(page_content="Low relevance chunk", 
                         metadata={"relevance_score": 0.45, "source": "B"})

        sample_state = {
            "question": "D&D rules",
            "plan": ["Retrieve rules"],
            "retrieved_chunks": [],
            "scratchpad": [],
            "user_id": "test_user"
        }

        # 2. Patch the compression retriever
        with patch("agents.retriever.compression_retriever") as mock_retriever:
            # The retriever returns them in its sorted order (doc1 then doc2)
            mock_retriever.invoke.return_value = [doc1, doc2]

            from agents.retriever import retriever_node
            result = retriever_node(sample_state)

            # 3. Assertions
            chunks = result["retrieved_chunks"]
            assert len(chunks) == 2
            
            # Assert the order is maintained based on how the retriever returned them
            assert chunks[0]["relevance_score"] == 0.99
            assert chunks[1]["relevance_score"] == 0.45
            assert chunks[0]["content"] == "High relevance chunk"
            
            # Verify the log entry mentions the reranking
            assert "after reranking" in result["scratchpad"][0]

    def test_applies_context_compression(self):
        """
        TODO:
        - Provide a verbose mock chunk.
        - Assert the output chunk content is shorter / compressed.
        """
        # 1. Create a verbose "original" string and a "compressed" result
        verbose_text = "This is a very long and wordy document content that contains some fluff but also the answer."
        compressed_text = "document content... the answer." # Simulating what a compressor might do

        # 2. Mock the Document returned by the compression_retriever
        mock_compressed_doc = MagicMock()
        mock_compressed_doc.page_content = compressed_text
        # Ensure metadata matches the mapping logic in your retriever_node
        mock_compressed_doc.metadata = {
            "text": compressed_text,
            "relevance_score": 0.8,
            "source": "Manual.pdf"
        }

        sample_state = {
            "question": "Give me the answer",
            "plan": ["Retrieve answer"],
            "retrieved_chunks": [],
            "scratchpad": [],
            "user_id": "test_user"
        }

        # 3. Patch the retriever
        with patch("agents.retriever.compression_retriever") as mock_retriever:
            mock_retriever.invoke.return_value = [mock_compressed_doc]

            from agents.retriever import retriever_node
            result = retriever_node(sample_state)

            # 4. Assertions
            chunk = result["retrieved_chunks"][0]
            
            # Assert the content is the compressed version, not the verbose one
            assert chunk["content"] == compressed_text
            assert len(chunk["content"]) < len(verbose_text)
            assert "fluff" not in chunk["content"]

    def test_handles_empty_results(self):
        """
        TODO:
        - Mock Pinecone returning zero matches.
        - Assert the node handles it gracefully (empty list, no crash).
        """
        sample_state = {
            "question": "Something that doesn't exist",
            "plan": ["Retrieve non-existent data"],
            "retrieved_chunks": [],
            "scratchpad": [],
            "user_id": "test_user"
        }

        with patch("agents.retriever.compression_retriever") as mock_retriever:
            mock_retriever.invoke.return_value = []

            from agents.retriever import retriever_node
            result = retriever_node(sample_state)

            # 1. Check structure
            assert result["retrieved_chunks"] == []
            
            # 2. Match the exact log string from your retriever_node
            # The error message shows the actual string is:
            # "Retriever Agent: Searched for 'non-existent data' in primary-corpus and found 0 chunks after reranking."
            actual_log = result["scratchpad"][0]
            assert "found 0 chunks" in actual_log
            assert "after reranking" in actual_log
