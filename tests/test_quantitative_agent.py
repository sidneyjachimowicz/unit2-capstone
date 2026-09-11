"""
test_quantitative_agent.py

Two kinds of tests here:
1. Offline tests against the real seeded SQLite database (schema
   introspection, SQL extraction, execution) -- zero API calls.
2. Fully mocked tests of the Gemini-dependent pipeline (validation +
   retry logic) using unittest.mock, so the retry-on-rejection behavior
   is verified without spending any real API quota.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.quantitative_agent import (
    get_schema_description,
    _extract_sql,
    _execute_query,
    answer_quantitative_question,
    is_unusual_query,
)


# --- Offline tests (real DB, no API) ---------------------------------

def test_schema_description_excludes_internal_tables():
    schema = get_schema_description()
    assert "sqlite_sequence" not in schema


def test_schema_description_includes_sample_values():
    """Regression test for the Q4/date-format bug fix: schema must show
    a real sample value per column, not just column names/types."""
    schema = get_schema_description()
    assert "e.g." in schema
    assert "revenue" in schema
    assert "customers" in schema
    assert "tickets" in schema
    assert "expenses" in schema


def test_schema_description_shows_all_category_values_for_low_cardinality_columns():
    """
    Regression test for the 'pull_request' bug: the ticket 'type' column
    has only 4 distinct values, but a policy doc discussing "pull requests"
    could mislead Gemini into guessing that literal string as the stored
    value. The schema must show ALL valid values for such columns so
    Gemini can match the question's wording to the actual stored value.
    """
    schema = get_schema_description()
    assert "code_review" in schema
    assert "bug_fix" in schema
    # The exact wrong value that caused the real bug should never need
    # to be guessed, since the real valid values are spelled out.
    assert "pull_request" not in schema


def test_extract_sql_from_markdown_fence():
    text = "```sql\nSELECT * FROM customers\n```"
    assert _extract_sql(text) == "SELECT * FROM customers"


def test_extract_sql_plain_text():
    text = "SELECT * FROM customers"
    assert _extract_sql(text) == "SELECT * FROM customers"


def test_execute_query_against_real_db():
    rows = _execute_query("SELECT COUNT(*) as cnt FROM customers")
    assert len(rows) == 1
    assert rows[0]["cnt"] > 0


# --- Mocked pipeline tests (no real API calls) ------------------------

@patch("src.quantitative_agent._ask_gemini_for_sql")
def test_answer_pipeline_valid_query_first_try(mock_ask):
    """If Gemini's first query is already valid, no retry should occur."""
    mock_ask.return_value = ("SELECT COUNT(*) FROM customers WHERE status='active'", {})

    result = answer_quantitative_question("How many active customers do we have?")

    assert result["success"] is True
    assert mock_ask.call_count == 1  # no retry needed


@patch("src.quantitative_agent._ask_gemini_for_sql")
def test_answer_pipeline_retries_once_on_rejection(mock_ask):
    """If the first query is rejected, the agent should retry exactly
    once with feedback, then succeed if the retry is valid."""
    mock_ask.side_effect = [
        ("DROP TABLE customers", {}),              # first attempt: rejected
        ("SELECT COUNT(*) FROM customers WHERE status='active'", {}),    # retry: valid
    ]

    result = answer_quantitative_question("How many active customers do we have?")

    assert result["success"] is True
    assert mock_ask.call_count == 2  # exactly one retry


@patch("src.quantitative_agent._ask_gemini_for_sql")
def test_answer_pipeline_fails_after_two_rejections(mock_ask):
    """If both the original and the retry are rejected, the agent must
    give up rather than executing anything against the database."""
    mock_ask.side_effect = [
        ("DROP TABLE customers", {}),
        ("DELETE FROM customers", {}),
    ]

    result = answer_quantitative_question("How many customers do we have?")

    assert result["success"] is False
    assert "rejected twice" in result["error"]
    assert mock_ask.call_count == 2  # never a third attempt


# --- Human-in-the-loop confirmation for unusual queries (Gold) ---------

def test_is_unusual_query_flags_missing_where():
    assert is_unusual_query("SELECT * FROM customers") is True


def test_is_unusual_query_allows_query_with_where():
    assert is_unusual_query("SELECT * FROM customers WHERE status='active'") is False


def test_is_unusual_query_case_insensitive():
    assert is_unusual_query("select * from customers where id=1") is False


@patch("src.quantitative_agent._ask_gemini_for_sql")
def test_unusual_query_declined_by_user_does_not_execute(mock_ask):
    """If the user declines confirmation on an unusual (no-WHERE) query,
    it must NOT be executed against the database."""
    mock_ask.return_value = ("SELECT * FROM customers", {})

    result = answer_quantitative_question(
        "everyone", confirm_callback=lambda q: False
    )

    assert result["success"] is False
    assert result.get("unusual_query_declined") is True
    assert "results" not in result


@patch("src.quantitative_agent._ask_gemini_for_sql")
def test_unusual_query_accepted_by_user_executes_normally(mock_ask):
    """If the user confirms, the unusual query proceeds and executes."""
    mock_ask.return_value = ("SELECT COUNT(*) as cnt FROM customers", {})

    result = answer_quantitative_question(
        "everyone", confirm_callback=lambda q: True
    )

    assert result["success"] is True
    assert result["results"][0]["cnt"] > 0


@patch("src.quantitative_agent._ask_gemini_for_sql")
def test_normal_query_never_triggers_confirmation(mock_ask):
    """A query WITH a WHERE clause should never even call the
    confirmation callback -- confirmation is only for unusual queries."""
    mock_ask.return_value = ("SELECT COUNT(*) as cnt FROM customers WHERE status='active'", {})

    call_count = {"n": 0}
    def tracking_confirm(q):
        call_count["n"] += 1
        return True

    result = answer_quantitative_question("active customers", confirm_callback=tracking_confirm)

    assert result["success"] is True
    assert call_count["n"] == 0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])