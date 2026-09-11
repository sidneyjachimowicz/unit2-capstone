"""
quantitative_agent.py

Converts natural language questions into SQL against the local SQLite
database, validates every generated query before execution, and returns
results. If a generated query fails validation, the agent sends the
rejection reason back to Gemini and asks it to try once more before
giving up.
"""

import os
import sqlite3
import re
from dotenv import load_dotenv
from google import genai

from src.sql_validator import validate_sql
from src.tokenomics import log_usage
from src.gemini_utils import generate_with_retry

load_dotenv()

DB_PATH = os.path.join("data", "enterprise.db")
MODEL_NAME = "gemini-3.6-flash"
AGENT_NAME = "QuantitativeAgent"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment.")
        _client = genai.Client(api_key=api_key)
    return _client


def get_schema_description(db_path: str = DB_PATH) -> str:
    """
    Introspect the SQLite database and build a plain-text schema
    description so Gemini knows what tables/columns exist.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
    )
    tables = [row[0] for row in cur.fetchall()]

    lines = []
    for table in tables:
        cur.execute(f"PRAGMA table_info({table})")
        columns = cur.fetchall()

        # Grab one sample row so Gemini can see actual data formats
        # (e.g. that 'month' is stored as 'YYYY-MM', not 'Q4' or 'October').
        cur.execute(f"SELECT * FROM {table} LIMIT 1")
        sample_row = cur.fetchone()

        col_descs = []
        for i, col in enumerate(columns):
            col_name, col_type = col[1], col[2]
            sample_val = sample_row[i] if sample_row else "N/A"

            # For TEXT columns, check if this looks like a category/enum
            # column (few distinct values) and show ALL valid values, not
            # just one sample. A single sample value isn't enough for
            # Gemini to know the exact stored spelling when a question's
            # wording doesn't match it (e.g. "pull request" in a policy
            # doc vs. the actual stored value "code_review").
            if col_type.upper() == "TEXT":
                cur.execute(f"SELECT DISTINCT {col_name} FROM {table} LIMIT 11")
                distinct_vals = [row[0] for row in cur.fetchall()]
                if 1 < len(distinct_vals) <= 10:
                    col_descs.append(
                        f"{col_name} ({col_type}, valid values: {distinct_vals!r})"
                    )
                    continue

            col_descs.append(f"{col_name} ({col_type}, e.g. {sample_val!r})")

        lines.append(f"- {table}: {', '.join(col_descs)}")

    conn.close()
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    """
    Strip markdown code fences if Gemini wraps the query in ```sql ... ```.
    """
    text = text.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _ask_gemini_for_sql(question: str, schema: str, feedback: str = None) -> tuple[str, dict]:
    """
    Send the question (and optional prior-rejection feedback) to Gemini,
    return the raw generated SQL string and usage metadata.
    """
    client = _get_client()

    prompt = f"""You are a SQL generation assistant. Given a database schema and a
natural language question, write a single SQLite SELECT query that answers it.

Schema:
{schema}

Rules:
- Only generate SELECT statements. Never generate DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
- Return ONLY the SQL query, no explanation, no markdown formatting.

Question: {question}
"""

    if feedback:
        prompt += f"\n\nYour previous query was rejected for this reason: {feedback}\nPlease generate a corrected SELECT query."

    response = generate_with_retry(client, MODEL_NAME, prompt)

    usage = response.usage_metadata
    log_usage(
        AGENT_NAME,
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )

    return _extract_sql(response.text), usage


def _execute_query(query: str, db_path: str = DB_PATH) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def is_unusual_query(query: str) -> bool:
    """
    Flags a validated query as 'unusual' and worth a human's eyes before
    executing -- specifically, a SELECT with no WHERE clause, which could
    return or aggregate across every row in a table unintentionally.
    This runs only on queries that already passed validate_sql(); it's a
    judgment/safety nudge, not a security gate.
    """
    query_upper = query.upper()
    return "WHERE" not in query_upper


def _default_confirm(query: str) -> bool:
    """
    Default human-in-the-loop confirmation: prompt on the CLI. Tests and
    non-interactive callers should pass their own confirm_callback instead
    of relying on this (e.g. a mock that always returns True/False).
    """
    print(f"\n[Confirmation needed] This query has no WHERE clause and will "
          f"run against the entire table:\n  {query}")
    response = input("Proceed anyway? [y/N]: ").strip().lower()
    return response == "y"


def answer_quantitative_question(
    question: str,
    db_path: str = DB_PATH,
    confirm_callback=None,
) -> dict:
    """
    Full pipeline: NL question -> Gemini SQL -> validate -> (retry once if
    rejected) -> [human confirmation if unusual] -> execute -> return
    structured result.

    confirm_callback: a function taking the SQL query string and returning
    True (proceed) or False (abort). Defaults to an interactive CLI prompt.
    Pass a custom callback (e.g. a test mock, or a non-interactive "auto
    reject" function) to control this without needing real user input.
    """
    if confirm_callback is None:
        confirm_callback = _default_confirm

    schema = get_schema_description(db_path)

    sql_query, _ = _ask_gemini_for_sql(question, schema)
    validation = validate_sql(sql_query)

    if not validation["valid"]:
        # One retry, giving Gemini the rejection reason as feedback.
        sql_query_retry, _ = _ask_gemini_for_sql(question, schema, feedback=validation["reason"])
        validation_retry = validate_sql(sql_query_retry)

        if not validation_retry["valid"]:
            return {
                "success": False,
                "question": question,
                "sql": sql_query_retry,
                "error": f"Query rejected twice. Last reason: {validation_retry['reason']}",
            }

        sql_query = sql_query_retry

    if is_unusual_query(sql_query):
        if not confirm_callback(sql_query):
            return {
                "success": False,
                "question": question,
                "sql": sql_query,
                "error": "Query flagged as unusual (no WHERE clause) and was not confirmed by the user.",
                "unusual_query_declined": True,
            }

    try:
        rows = _execute_query(sql_query, db_path)
    except sqlite3.Error as e:
        return {
            "success": False,
            "question": question,
            "sql": sql_query,
            "error": f"SQL execution error: {e}",
        }

    return {
        "success": True,
        "question": question,
        "sql": sql_query,
        "results": rows,
    }


if __name__ == "__main__":
    # Quick manual smoke test
    result = answer_quantitative_question("What's our customer churn rate?")
    print("\n--- RESULT ---")
    print(result)