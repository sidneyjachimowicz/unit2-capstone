"""
gemini_utils.py

Shared helper for calling Gemini with automatic retry on transient server
errors (e.g. 503 UNAVAILABLE during high demand). Used by all three agents
so a single transient failure doesn't kill a multi-step pipeline (the
Manager Agent alone can chain 5+ sequential Gemini calls for one complex
question).
"""

import time
from google.genai import errors as genai_errors


def generate_with_retry(client, model: str, contents: str, max_retries: int = 3, base_delay: float = 2.0):
    """
    Call client.models.generate_content with retry on transient server
    errors (5xx). Raises immediately on non-transient errors (e.g. 404,
    auth failures) since retrying those would just waste time.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except genai_errors.ServerError as e:
            last_error = e
            wait = base_delay * (2 ** attempt)
            print(f"  [retry] Gemini server error (attempt {attempt + 1}/{max_retries}), "
                  f"waiting {wait:.0f}s: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait)
    raise last_error