"""
gemini_utils.py

Shared helper for calling Gemini with automatic retry on transient server
errors (e.g. 503 UNAVAILABLE during high demand) and per-minute rate-limit
errors (429 RESOURCE_EXHAUSTED with a per-minute quota, which clears on
its own within roughly a minute -- unlike a per-day quota, which does not
and should not be retried). Used by all three agents so a single hiccup
doesn't kill a multi-step pipeline (the Manager Agent alone can chain 5+
sequential Gemini calls for one complex question).
"""

import time
from google.genai import errors as genai_errors


def _is_per_minute_rate_limit(error: genai_errors.ClientError) -> bool:
    """
    Distinguish a per-minute rate limit (worth retrying, clears quickly)
    from a per-day quota exhaustion (not worth retrying, won't clear for
    hours). Google's error body includes a "details" list that can
    contain MULTIPLE entries (e.g. a Help link AND a QuotaFailure object)
    -- the violations live inside whichever entry has a "violations" key,
    not necessarily the first one, so every entry must be checked.
    """
    try:
        details = error.details or {}
        detail_entries = details.get("error", {}).get("details", [])
        for entry in detail_entries:
            for v in entry.get("violations", []):
                if "PerMinute" in v.get("quotaId", ""):
                    return True
    except (AttributeError, KeyError, TypeError):
        pass
    return False


def generate_with_retry(client, model: str, contents: str, max_retries: int = 3, base_delay: float = 2.0):
    """
    Call client.models.generate_content with retry on:
    - Transient server errors (5xx / ServerError)
    - Per-minute rate limits (429 with a PerMinute quotaId) -- these
      clear on their own, so a ~60s wait and retry is worthwhile.

    Does NOT retry per-day quota exhaustion (429 with a PerDay quotaId)
    or other client errors (e.g. 404, auth failures), since retrying
    those just wastes time without changing the outcome.
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
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429 and _is_per_minute_rate_limit(e):
                last_error = e
                wait = 65  # per-minute quotas reset roughly every 60s
                print(f"  [retry] Per-minute rate limit hit (attempt {attempt + 1}/{max_retries}), "
                      f"waiting {wait}s for quota reset: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait)
            else:
                # Per-day quota exhaustion or other client error: don't
                # waste time retrying, fail fast with a clear message.
                raise
    raise last_error