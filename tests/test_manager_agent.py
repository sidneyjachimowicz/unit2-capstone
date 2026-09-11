"""
test_manager_agent.py

Fully mocked tests of the Manager Agent's routing, decomposition,
synthesis, and completeness-check-with-follow-up logic. No real Gemini
calls, no real database/vector store access -- everything the Manager
depends on is mocked so these tests run instantly and free of charge.
"""

import sys
import os
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.manager_agent import (
    classify_question,
    _extract_json,
    handle_question,
)


# --- JSON extraction (pure logic, no API) ------------------------------

def test_extract_json_plain():
    result = _extract_json('{"classification": "complex", "reasoning": "test"}')
    assert result["classification"] == "complex"


def test_extract_json_with_markdown_fence():
    result = _extract_json('```json\n{"classification": "qualitative", "reasoning": "x"}\n```')
    assert result["classification"] == "qualitative"


# --- Classification (mocked Gemini) ------------------------------------

@patch("src.manager_agent._call_gemini")
def test_classify_returns_parsed_json(mock_call):
    mock_call.return_value = '{"classification": "qualitative", "reasoning": "policy only"}'
    result = classify_question("What is our security policy?")
    assert result["classification"] == "qualitative"


@patch("src.manager_agent._call_gemini")
def test_classify_fails_open_to_complex_on_parse_error(mock_call):
    """If Gemini's classifier response can't be parsed as JSON, the
    Manager should default to 'complex' rather than crash or silently
    under-answer -- complex triggers both agents, the safer failure mode."""
    mock_call.return_value = "not valid json at all"
    result = classify_question("Some question")
    assert result["classification"] == "complex"


# --- Full routing via handle_question (all dependencies mocked) -------

@patch("src.manager_agent.classify_question")
@patch("src.manager_agent.answer_qualitative_question")
def test_handle_question_routes_qualitative(mock_qual, mock_classify):
    mock_classify.return_value = {"classification": "qualitative", "reasoning": "policy question"}
    mock_qual.return_value = {"answer": "MFA required.", "sources": [{"doc_name": "security_policy", "section": "Password & Authentication"}]}

    result = handle_question("What's our MFA policy?")

    assert result["answer"] == "MFA required."
    mock_qual.assert_called_once()


@patch("src.manager_agent.classify_question")
@patch("src.manager_agent.answer_quantitative_question")
def test_handle_question_routes_quantitative(mock_quant, mock_classify):
    mock_classify.return_value = {"classification": "quantitative", "reasoning": "needs DB numbers"}
    mock_quant.return_value = {"success": True, "results": [{"churn": 0.2}], "sql": "SELECT ..."}

    result = handle_question("What's our churn rate?")

    assert result["answer"] == [{"churn": 0.2}]
    mock_quant.assert_called_once()


@patch("src.manager_agent.classify_question")
@patch("src.manager_agent._decompose_complex_question")
@patch("src.manager_agent._synthesize_and_check")
def test_handle_question_complex_no_followup_needed(mock_synth, mock_decompose, mock_classify):
    """When the first synthesis pass reports itself complete, no
    follow-up call should be made."""
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
    mock_synth.assert_called_once()  # no second synthesis call


@patch("src.manager_agent.classify_question")
@patch("src.manager_agent._decompose_complex_question")
@patch("src.manager_agent._synthesize_and_check")
@patch("src.manager_agent.answer_quantitative_question")
def test_handle_question_complex_triggers_followup(mock_quant_retry, mock_synth, mock_decompose, mock_classify):
    """
    This is the key test for the spec's required 'clarifying follow-up
    if the initial routing produces an incomplete answer' behavior.
    When the first synthesis reports itself incomplete, the Manager must
    re-query quantitative with the gap as context and re-synthesize --
    exactly once, not in an infinite loop.
    """
    mock_classify.return_value = {"classification": "complex", "reasoning": "needs both"}
    mock_decompose.return_value = (
        {"answer": "Standard is 48 hours", "sources": []},
        {"success": True, "sql": "SELECT AVG(turnaround_hours)...", "results": [{"avg": 96}]},
        "What is the average code review turnaround time?",
    )
    # First synthesis call: incomplete (e.g. averaged across all ticket
    # types instead of just code_review)
    # Second synthesis call (after follow-up): complete
    mock_synth.side_effect = [
        {"answer": "Partial answer", "complete": False, "gap": "Need to filter by code_review type specifically"},
        {"answer": "79% of code reviews meet the 48-hour standard.", "complete": True, "gap": ""},
    ]
    mock_quant_retry.return_value = {"success": True, "sql": "SELECT ... WHERE type='code_review'", "results": [{"pct": 0.79}]}

    result = handle_question("Are our code review turnaround times meeting the standard?")

    assert result["followup_used"] is True
    assert result["answer"] == "79% of code reviews meet the 48-hour standard."
    assert mock_synth.call_count == 2       # exactly one retry, not more
    assert mock_quant_retry.call_count == 1  # the follow-up quantitative call


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])