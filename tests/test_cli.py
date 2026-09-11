"""
test_cli.py

Tests for the CLI's output formatting logic. Pure string formatting,
no API calls, no user input required.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cli import format_result, EXIT_COMMANDS


def test_format_qualitative_result():
    result = {
        "classification": {"classification": "qualitative", "reasoning": "policy question"},
        "answer": "MFA is required.",
        "sources": [{"doc_name": "security_policy", "section": "Password & Authentication"}],
    }
    output = format_result(result)
    assert "Classified as: qualitative" in output
    assert "MFA is required." in output
    assert "security_policy" in output
    assert "Password & Authentication" in output


def test_format_quantitative_result_includes_sql():
    result = {
        "classification": {"classification": "quantitative", "reasoning": "numeric"},
        "answer": [{"count": 40}],
        "sql": "SELECT COUNT(*) FROM expenses WHERE amount > 500",
    }
    output = format_result(result)
    assert "SELECT COUNT(*)" in output


def test_format_complex_result_shows_followup_note():
    result = {
        "classification": {"classification": "complex", "reasoning": "needs both"},
        "answer": "79% meet the standard.",
        "sources": [{"doc_name": "code_review_process", "section": "Process Steps"}],
        "sql": "SELECT ...",
        "followup_used": True,
    }
    output = format_result(result)
    assert "clarifying follow-up" in output


def test_format_result_without_followup_omits_note():
    result = {
        "classification": {"classification": "complex", "reasoning": "needs both"},
        "answer": "Some answer.",
        "sources": [],
        "sql": None,
        "followup_used": False,
    }
    output = format_result(result)
    assert "clarifying follow-up" not in output


def test_exit_commands_recognized():
    assert "exit" in EXIT_COMMANDS
    assert "quit" in EXIT_COMMANDS
    assert "q" in EXIT_COMMANDS


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])