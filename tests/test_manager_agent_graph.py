"""
test_manager_agent_graph.py

Tests for the LangGraph refactor of the Manager Agent's routing logic
(Silver stretch goal). Verifies the graph structure itself, plus the
same routing/decomposition/follow-up behaviors already proven in
test_manager_agent.py, now exercised through the graph's execution path
instead of the original hand-written if/else chain.
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.manager_agent_graph import build_graph, handle_question


# --- Graph structure ----------------------------------------------------

def test_graph_compiles():
    graph = build_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    expected = {"classify", "qualitative", "quantitative", "decompose", "synthesize", "followup"}
    assert expected.issubset(nodes)


def test_graph_has_conditional_routing_from_classify():
    graph = build_graph()
    edges = graph.get_graph().edges
    classify_edges = [e for e in edges if e.source == "classify"]
    targets = {e.target for e in classify_edges}
    assert targets == {"qualitative", "quantitative", "decompose"}
    assert all(e.conditional for e in classify_edges)


def test_graph_has_conditional_routing_from_synthesize():
    graph = build_graph()
    edges = graph.get_graph().edges
    synth_edges = [e for e in edges if e.source == "synthesize"]
    targets = {e.target for e in synth_edges}
    assert "followup" in targets
    assert all(e.conditional for e in synth_edges)


# --- Routing behavior (mocked, no API calls) ---------------------------

@patch("src.manager_agent_graph.answer_qualitative_question")
@patch("src.manager_agent_graph.classify_question")
def test_graph_routes_qualitative(mock_classify, mock_qual):
    mock_classify.return_value = {"classification": "qualitative", "reasoning": "policy only"}
    mock_qual.return_value = {
        "answer": "MFA is required.",
        "sources": [{"doc_name": "security_policy", "section": "Password & Authentication"}],
    }

    result = handle_question("What's our MFA policy?")

    assert result["answer"] == "MFA is required."
    mock_qual.assert_called_once()


@patch("src.manager_agent_graph.answer_quantitative_question")
@patch("src.manager_agent_graph.classify_question")
def test_graph_routes_quantitative(mock_classify, mock_quant):
    mock_classify.return_value = {"classification": "quantitative", "reasoning": "needs DB numbers"}
    mock_quant.return_value = {"success": True, "results": [{"churn": 0.2}], "sql": "SELECT ..."}

    result = handle_question("What's our churn rate?")

    assert result["answer"] == [{"churn": 0.2}]
    mock_quant.assert_called_once()


@patch("src.manager_agent_graph._synthesize_and_check")
@patch("src.manager_agent_graph._decompose_complex_question")
@patch("src.manager_agent_graph.classify_question")
def test_graph_complex_no_followup_needed(mock_classify, mock_decompose, mock_synth):
    mock_classify.return_value = {"classification": "complex", "reasoning": "needs both"}
    mock_decompose.return_value = (
        {"answer": "Threshold is $500", "sources": []},
        {"success": True, "sql": "SELECT COUNT(*)...", "results": [{"count": 40}]},
        "How many expenses over $500?",
    )
    mock_synth.return_value = {"answer": "40 expenses exceeded $500.", "complete": True, "gap": ""}

    result = handle_question("Complex red-herring question")

    assert result["answer"] == "40 expenses exceeded $500."
    assert result["followup_used"] is False
    mock_synth.assert_called_once()


@patch("src.manager_agent_graph.answer_quantitative_question")
@patch("src.manager_agent_graph._synthesize_and_check")
@patch("src.manager_agent_graph._decompose_complex_question")
@patch("src.manager_agent_graph.classify_question")
def test_graph_complex_triggers_followup_exactly_once(mock_classify, mock_decompose, mock_synth, mock_quant_retry):
    """
    The critical Silver/Gold test: the graph must reproduce the same
    'exactly one clarifying follow-up' behavior as the original
    hand-written Manager Agent, not loop indefinitely.
    """
    mock_classify.return_value = {"classification": "complex", "reasoning": "needs both"}
    mock_decompose.return_value = (
        {"answer": "Standard is 48 hours", "sources": []},
        {"success": True, "sql": "SELECT AVG(turnaround_hours)...", "results": [{"avg": 96}]},
        "What is the average code review turnaround time?",
    )
    mock_synth.side_effect = [
        {"answer": "Partial answer", "complete": False, "gap": "Need to filter by code_review type"},
        {"answer": "79% of code reviews meet the 48-hour standard.", "complete": True, "gap": ""},
    ]
    mock_quant_retry.return_value = {"success": True, "sql": "SELECT ... WHERE type='code_review'", "results": [{"pct": 0.79}]}

    result = handle_question("Are our code review turnaround times meeting the standard?")

    assert result["followup_used"] is True
    assert result["answer"] == "79% of code reviews meet the 48-hour standard."
    assert mock_synth.call_count == 2       # exactly one retry, not more
    assert mock_quant_retry.call_count == 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])