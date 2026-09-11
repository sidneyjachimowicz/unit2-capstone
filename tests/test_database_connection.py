"""
test_database_connection.py

Verifies the SQLite database exists, has the expected schema, and
contains data that supports the required query types. No API calls.
Run `python scripts/seed_data.py` first if this fails with "database
not found".
"""

import sys
import os
import sqlite3
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

DB_PATH = os.path.join("data", "enterprise.db")


@pytest.fixture(scope="module")
def db_connection():
    if not os.path.exists(DB_PATH):
        pytest.skip(f"{DB_PATH} not found -- run scripts/seed_data.py first")
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


def test_database_file_exists():
    assert os.path.exists(DB_PATH), "Run scripts/seed_data.py to create the database"


def test_expected_tables_exist(db_connection):
    cur = db_connection.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    expected = {"revenue", "customers", "tickets", "expenses"}
    assert expected.issubset(tables)


def test_all_tables_have_data(db_connection):
    cur = db_connection.cursor()
    for table in ["revenue", "customers", "tickets", "expenses"]:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        count = cur.fetchone()[0]
        assert count > 0, f"{table} has no rows"


def test_customers_have_churn_data(db_connection):
    """Sanity check that churn rate is actually computable (not 0% or 100%)."""
    cur = db_connection.cursor()
    cur.execute("SELECT COUNT(*) FROM customers WHERE status = 'churned'")
    churned = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM customers")
    total = cur.fetchone()[0]
    assert 0 < churned < total


def test_expenses_span_the_approval_threshold(db_connection):
    """Sanity check that the $500 threshold from the policy doc actually
    has real data on both sides of it -- otherwise the red-herring query
    would have a trivial (0 or 100%) answer."""
    cur = db_connection.cursor()
    cur.execute("SELECT COUNT(*) FROM expenses WHERE amount > 500")
    above = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM expenses WHERE amount <= 500")
    below = cur.fetchone()[0]
    assert above > 0 and below > 0


def test_code_review_tickets_span_the_turnaround_standard(db_connection):
    """Sanity check that the 48-hour standard from the policy doc has
    real data on both sides of it for code_review tickets specifically."""
    cur = db_connection.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM tickets WHERE type='code_review' AND turnaround_hours <= 48"
    )
    within = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM tickets WHERE type='code_review' AND turnaround_hours > 48"
    )
    outside = cur.fetchone()[0]
    assert within > 0 and outside > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])