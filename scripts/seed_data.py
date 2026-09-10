"""
seed_data.py

Generates a small synthetic SQLite database for the Quantitative Agent.
Run this once to create data/enterprise.db with realistic sample data
covering all required query types from the capstone spec.
"""

import sqlite3
import random
from datetime import datetime, timedelta
import os

random.seed(42)  # reproducible data

DB_PATH = os.path.join("data", "enterprise.db")
REGIONS = ["Northeast", "Southeast", "Midwest", "West"]
DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Operations", "HR"]
TICKET_TYPES = ["code_review", "bug_fix", "feature_request", "support"]


def create_schema(conn):
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS revenue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            month TEXT NOT NULL,        -- format YYYY-MM
            region TEXT NOT NULL,
            amount REAL NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            signup_date TEXT NOT NULL,
            cancellation_date TEXT,      -- NULL if still active
            status TEXT NOT NULL         -- 'active' or 'churned'
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            closed_at TEXT,               -- NULL if still open
            turnaround_hours REAL         -- precomputed for convenience
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department TEXT NOT NULL,
            amount REAL NOT NULL,
            submitted_date TEXT NOT NULL,
            description TEXT
        )
    """)

    conn.commit()


def seed_revenue(conn, months_back=12):
    cur = conn.cursor()
    today = datetime.today()
    rows = []
    for m in range(months_back):
        month_date = today - timedelta(days=30 * m)
        month_str = month_date.strftime("%Y-%m")
        for region in REGIONS:
            # slight upward trend over time + region variance
            base = 80000 + (months_back - m) * 1500
            region_factor = {"Northeast": 1.2, "Southeast": 0.9, "Midwest": 1.0, "West": 1.1}[region]
            amount = round(base * region_factor * random.uniform(0.85, 1.15), 2)
            rows.append((month_str, region, amount))

    cur.executemany(
        "INSERT INTO revenue (month, region, amount) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    print(f"Seeded {len(rows)} revenue rows")


def seed_customers(conn, n=200):
    cur = conn.cursor()
    today = datetime.today()
    rows = []
    for _ in range(n):
        signup = today - timedelta(days=random.randint(30, 730))
        # ~22% churn rate baked in deliberately
        if random.random() < 0.22:
            status = "churned"
            cancel = signup + timedelta(days=random.randint(15, 500))
            if cancel > today:
                cancel = today
            cancel_str = cancel.strftime("%Y-%m-%d")
        else:
            status = "active"
            cancel_str = None
        rows.append((signup.strftime("%Y-%m-%d"), cancel_str, status))

    cur.executemany(
        "INSERT INTO customers (signup_date, cancellation_date, status) VALUES (?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Seeded {len(rows)} customer rows")


def seed_tickets(conn, n=150):
    """
    Code review tickets are deliberately given a mix of turnaround times,
    some within a 48-hour policy commitment and some outside it, so the
    'harder query' about meeting the documented standard has a real,
    checkable answer (~70% meet the standard, ~30% miss it).
    """
    cur = conn.cursor()
    today = datetime.today()
    rows = []
    for _ in range(n):
        ttype = random.choice(TICKET_TYPES)
        opened = today - timedelta(days=random.randint(1, 180), hours=random.randint(0, 23))

        if ttype == "code_review":
            # 70% within 48 hours, 30% blown past it
            if random.random() < 0.70:
                turnaround = random.uniform(2, 47)
            else:
                turnaround = random.uniform(49, 120)
        else:
            turnaround = random.uniform(4, 96)

        closed = opened + timedelta(hours=turnaround)
        if closed > today:
            closed = None
            turnaround_val = None
            closed_str = None
        else:
            closed_str = closed.strftime("%Y-%m-%d %H:%M:%S")
            turnaround_val = round(turnaround, 2)

        rows.append((ttype, opened.strftime("%Y-%m-%d %H:%M:%S"), closed_str, turnaround_val))

    cur.executemany(
        "INSERT INTO tickets (type, opened_at, closed_at, turnaround_hours) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Seeded {len(rows)} ticket rows")


def seed_expenses(conn, n=120):
    """
    Amounts are deliberately spread across a $500 approval threshold
    (documented in the qualitative policy doc) so the red-herring
    'expense sign-off' query has a real, checkable answer.
    """
    cur = conn.cursor()
    today = datetime.today()
    rows = []
    descriptions = [
        "Software license", "Travel reimbursement", "Client dinner",
        "Office supplies", "Conference registration", "Equipment purchase",
        "Contractor invoice", "Training course",
    ]
    for _ in range(n):
        dept = random.choice(DEPARTMENTS)
        # ~35% of requests land above the $500 approval threshold
        if random.random() < 0.35:
            amount = round(random.uniform(500.01, 5000), 2)
        else:
            amount = round(random.uniform(10, 499.99), 2)
        submitted = today - timedelta(days=random.randint(1, 90))
        desc = random.choice(descriptions)
        rows.append((dept, amount, submitted.strftime("%Y-%m-%d"), desc))

    cur.executemany(
        "INSERT INTO expenses (department, amount, submitted_date, description) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    print(f"Seeded {len(rows)} expense rows")


def main():
    os.makedirs("data", exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Removed existing {DB_PATH} to reseed fresh")

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    seed_revenue(conn)
    seed_customers(conn)
    seed_tickets(conn)
    seed_expenses(conn)
    conn.close()

    print(f"\nDatabase created at {DB_PATH}")


if __name__ == "__main__":
    main()