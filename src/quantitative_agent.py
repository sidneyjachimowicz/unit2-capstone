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
        col_desc = ", ".join(f"{col[1]} ({col[2]})" for col in columns)
        lines.append(f"- {table}: {col_desc}")

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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

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


def answer_quantitative_question(question: str, db_path: str = DB_PATH) -> dict:
    """
    Full pipeline: NL question -> Gemini SQL -> validate -> (retry once if
    rejected) -> execute -> return structured result.
    """
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