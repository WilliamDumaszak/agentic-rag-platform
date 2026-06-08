"""
Unit tests for the LangGraph agent nodes.
All external dependencies (LLM, embeddings, retriever, web search) are mocked.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.nodes import decision_node, generate_node, rank_node, retrieve_node, web_search_node
from agent.state import AgentState


# ── decision node ─────────────────────────────────────────────────────────────

def test_decision_web_search_keyword():
    state = AgentState(query="Pesquise sobre novidades em logística")
    result = decision_node(state)
    assert result.next_step == "web_search"


def test_decision_retrieve_question():
    state = AgentState(query="What are the phases of supply chain?")
    result = decision_node(state)
    assert result.next_step == "retrieve"


def test_decision_generate_short():
    state = AgentState(query="Oi")
    result = decision_node(state)
    assert result.next_step == "generate"


# ── retrieve node ─────────────────────────────────────────────────────────────

@patch("agent.nodes._build_retriever")
def test_retrieve_node(mock_build_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Supply chain involves multiple stages.")
    ]
    mock_build_retriever.return_value = mock_retriever

    state = AgentState(query="supply chain phases")
    result = retrieve_node(state)

    mock_retriever.invoke.assert_called_once_with("supply chain phases")
    assert len(result.retrieved_docs) == 1
    assert result.source == "retrieval"


# ── web search node ───────────────────────────────────────────────────────────

@patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper")
def test_web_search_node(mock_search_cls):
    mock_search = MagicMock()
    mock_search.run.return_value = "Latest news about logistics..."
    mock_search_cls.return_value = mock_search

    state = AgentState(query="latest logistics news")
    result = web_search_node(state)

    assert result.ranked_response == "Latest news about logistics..."
    assert result.confidence_score == 1.0
    assert result.source == "web"


@patch("langchain_community.utilities.DuckDuckGoSearchAPIWrapper")
@patch("agent.nodes._build_retriever")
def test_web_search_fallback_to_retrieve(mock_build_retriever, mock_search_cls):
    mock_search = MagicMock()
    mock_search.run.side_effect = Exception("network error")
    mock_search_cls.return_value = mock_search

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [Document(page_content="Fallback doc")]
    mock_build_retriever.return_value = mock_retriever

    state = AgentState(query="latest news")
    result = web_search_node(state)
    assert result.source == "retrieval"


# ── generate node ─────────────────────────────────────────────────────────────

@patch("agent.nodes._build_llm")
@patch("agent.nodes._build_embeddings")
def test_generate_node(mock_emb, mock_llm_builder):
    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(return_value=mock_llm)
    mock_llm_builder.return_value = mock_llm

    with patch("agent.nodes.StrOutputParser") as mock_parser_cls:
        mock_parser = MagicMock()
        mock_parser_cls.return_value = mock_parser

        with patch("agent.nodes.PromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Generated answer"
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)
            mock_prompt_cls.from_template.return_value = mock_prompt

            state = AgentState(
                query="What is supply chain?",
                retrieved_docs=[Document(page_content="Supply chain context.")],
            )
            result = generate_node(state)
            # should have attempted to generate n_responses variants
            assert isinstance(result.possible_responses, list)


# ── rank node ─────────────────────────────────────────────────────────────────

@patch("agent.nodes._build_embeddings")
def test_rank_node_picks_best(mock_build_emb):
    mock_emb = MagicMock()
    mock_emb.embed_documents.side_effect = [
        [[0.1, 0.2, 0.3]],           # reference embedding
        [[0.1, 0.2, 0.3], [0.9, 0.0, 0.0]],  # response embeddings
    ]
    mock_build_emb.return_value = mock_emb

    state = AgentState(
        query="question",
        possible_responses=["response A", "response B"],
    )
    result = rank_node(state)
    assert result.ranked_response in ["response A", "response B"]
    assert 0.0 <= result.confidence_score <= 1.0


def test_rank_node_empty_responses():
    state = AgentState(query="question", possible_responses=[])
    result = rank_node(state)
    assert result.ranked_response == "No response generated."
    assert result.confidence_score == 0.0
