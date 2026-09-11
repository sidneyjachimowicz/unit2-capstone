"""
test_qualitative_agent.py

Offline tests for the markdown chunking logic (pure text processing, no
API/embedding calls), plus mocked tests of the retrieval + answer
generation pipeline using unittest.mock.
"""

import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.qualitative_agent import (
    _chunk_markdown_by_section,
    answer_qualitative_question,
)

SAMPLE_DOC = """# Sample Policy

## Overview
This is the overview section.

## Details
This is the details section with more content.

## Escalation
This is the escalation section.
"""


# --- Offline tests (pure text processing, no API) ---------------------

def test_chunking_produces_expected_sections():
    chunks = _chunk_markdown_by_section(SAMPLE_DOC, "sample_policy")
    sections = [c["section"] for c in chunks]
    assert sections == ["Overview", "Details", "Escalation"]


def test_chunking_folds_title_into_first_section():
    """Regression test for the generic-chunk retrieval bug fix: the doc
    title should NOT become its own standalone chunk."""
    chunks = _chunk_markdown_by_section(SAMPLE_DOC, "sample_policy")
    first_chunk_text = chunks[0]["text"]
    assert "Sample Policy" in first_chunk_text  # title folded in
    assert not any(c["section"] == "Title" for c in chunks)  # no standalone Title chunk


def test_chunking_no_empty_chunks():
    chunks = _chunk_markdown_by_section(SAMPLE_DOC, "sample_policy")
    for c in chunks:
        assert len(c["text"]) > 0


def test_chunking_attaches_doc_name():
    chunks = _chunk_markdown_by_section(SAMPLE_DOC, "sample_policy")
    for c in chunks:
        assert c["doc_name"] == "sample_policy"


def test_chunking_handles_doc_with_no_sections():
    """
    A doc with only a title and no ## headers should not silently produce
    zero chunks (which would make it unsearchable). Falls back to a
    single 'Full Document' chunk instead.
    """
    text = "# Just A Title\n\nSome intro text with no section headers at all."
    chunks = _chunk_markdown_by_section(text, "no_sections_doc")
    assert len(chunks) == 1
    assert chunks[0]["section"] == "Full Document"
    assert "Just A Title" in chunks[0]["text"]


# --- Mocked pipeline tests (no real API/embedding calls) --------------

@patch("src.qualitative_agent._get_gemini_client")
@patch("src.qualitative_agent._retrieve_relevant_chunks")
def test_answer_includes_source_attribution(mock_retrieve, mock_client):
    mock_retrieve.return_value = [
        {"text": "MFA is required.", "doc_name": "security_policy",
         "section": "Password & Authentication", "distance": 0.1}
    ]

    mock_response = MagicMock()
    mock_response.text = "MFA is required on all accounts."
    mock_response.usage_metadata.prompt_token_count = 50
    mock_response.usage_metadata.candidates_token_count = 10

    mock_gemini_client = MagicMock()
    mock_gemini_client.models.generate_content.return_value = mock_response
    mock_client.return_value = mock_gemini_client

    with patch("src.qualitative_agent.generate_with_retry", return_value=mock_response):
        result = answer_qualitative_question("What's our MFA policy?")

    assert result["success"] is True
    assert result["sources"][0]["doc_name"] == "security_policy"
    assert result["sources"][0]["section"] == "Password & Authentication"


@patch("src.qualitative_agent._retrieve_relevant_chunks")
def test_answer_handles_no_relevant_chunks(mock_retrieve):
    """If retrieval comes back empty, the agent must fail gracefully
    rather than sending an empty-context prompt to Gemini."""
    mock_retrieve.return_value = []

    result = answer_qualitative_question("Some totally unrelated question")

    assert result["success"] is False
    assert "No relevant documents found" in result["error"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])