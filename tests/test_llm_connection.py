"""
test_llm_connection.py

Automated LLM connection tests. Two kinds:
1. A mocked test verifying the Gemini client is constructed and called
   correctly -- runs every time, zero API cost, zero external dependency.
2. A real live connectivity check -- only runs if GEMINI_API_KEY is set,
   and is skipped (not failed) otherwise, so this suite still passes in
   CI or on a machine without API access/quota configured.
"""

import sys
import os
from unittest.mock import patch, MagicMock
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
load_dotenv()


def test_gemini_client_constructed_with_api_key():
    """Verify _get_client() builds a genai.Client using the API key from
    the environment, without making a real network call."""
    from src.manager_agent import _get_client
    import src.manager_agent as ma

    ma._client = None  # reset any cached client from other tests

    with patch("src.manager_agent.os.getenv", return_value="fake-test-key"), \
         patch("src.manager_agent.genai.Client") as mock_client_class:
        mock_client_class.return_value = MagicMock()
        client = _get_client()
        assert mock_client_class.called
        args, kwargs = mock_client_class.call_args
        assert kwargs.get("api_key") == "fake-test-key"

    ma._client = None  # clean up after ourselves


def test_missing_api_key_raises_clear_error():
    """If GEMINI_API_KEY isn't set, the agent should fail with a clear,
    actionable error rather than a confusing downstream exception."""
    from src.manager_agent import _get_client
    import src.manager_agent as ma

    ma._client = None

    with patch.dict(os.environ, {}, clear=True):
        with patch("src.manager_agent.os.getenv", return_value=None):
            with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
                _get_client()

    ma._client = None


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set -- skipping live connectivity check",
)
def test_live_gemini_connection():
    """
    Real connectivity smoke test. Uses exactly one live API call. Skips
    gracefully (does not fail the suite) if no key is configured, so this
    test file is safe to run in any environment.
    """
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Reply with exactly one word: 'connected'",
    )
    assert "connected" in response.text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])