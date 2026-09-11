# System Architecture

This document covers agent interactions and data flow in detail. For setup instructions and a high-level overview, see the main [README](../README.md).

## Component Overview

| Component | File | Responsibility |
|---|---|---|
| Manager Agent | `src/manager_agent.py` | Classifies questions, routes to the right agent(s), synthesizes multi-agent answers |
| Qualitative Agent | `src/qualitative_agent.py` | Semantic search over policy docs via Chroma + Sentence Transformers, answers via Gemini |
| Quantitative Agent | `src/quantitative_agent.py` | Natural language to SQL via Gemini, validated execution against SQLite |
| SQL Validator | `src/sql_validator.py` | Gatekeeps every generated query before it touches the database |
| Tokenomics | `src/tokenomics.py` | Logs token usage/cost for every Gemini call, across all three agents |
| Gemini Utils | `src/gemini_utils.py` | Shared retry logic for transient server errors and per-minute rate limits |
| CLI | `src/cli.py` | Interactive command-line entry point |

## Agent Interaction: Simple Qualitative Query

Example: *"What is our company's security policy?"*

```
User → CLI → Manager Agent
                 │
                 ├─ 1. classify_question()
                 │     Gemini call: reasons about the question, returns
                 │     {"classification": "qualitative", "reasoning": "..."}
                 │
                 └─ 2. answer_qualitative_question()
                       ├─ embed the question (Sentence Transformers, local, no API call)
                       ├─ query Chroma for top-3 relevant chunks
                       ├─ Gemini call: generate an answer grounded in those chunks
                       └─ return {answer, sources: [{doc_name, section}, ...]}
                 │
                 └─→ CLI displays answer + sources
```

Total Gemini calls: 2 (1 classify + 1 answer generation). Embedding is local and free.

## Agent Interaction: Simple Quantitative Query

Example: *"What's our customer churn rate?"*

```
User → CLI → Manager Agent
                 │
                 ├─ 1. classify_question()
                 │     Gemini call → {"classification": "quantitative", "reasoning": "..."}
                 │
                 └─ 2. answer_quantitative_question()
                       ├─ get_schema_description() -- reads live SQLite schema,
                       │   including sample values and full category lists for
                       │   low-cardinality columns (see "Known Limitations" in README)
                       ├─ Gemini call: question + schema → generated SQL
                       ├─ validate_sql(query) -- gate before execution
                       │     if REJECTED: Gemini call with rejection reason as
                       │     feedback → retry once → validate again → give up if
                       │     still rejected
                       ├─ execute query against data/enterprise.db
                       └─ return {success, sql, results}
                 │
                 └─→ CLI displays answer + SQL used
```

Total Gemini calls: 2 (1 classify + 1 SQL generation), or 3 if the validator rejects the first query and a retry is needed.

## Agent Interaction: Complex Query (Multi-Agent)

Example (the red-herring query): *"What's our policy on expense approvals, and how many expense requests last quarter would have required manager sign-off under that policy?"*

```
User → CLI → Manager Agent
                 │
                 ├─ 1. classify_question()
                 │     Gemini call: reasons about what data is ACTUALLY needed
                 │     to answer completely (not keyword matching) →
                 │     {"classification": "complex", "reasoning": "..."}
                 │
                 ├─ 2. _decompose_complex_question()
                 │     ├─ answer_qualitative_question(original_question)
                 │     │     → retrieves the $500 threshold from policy docs
                 │     │     (1 Gemini call internally)
                 │     └─ Gemini call: build a precise quantitative
                 │         sub-question using the retrieved policy fact
                 │         (e.g. "How many expenses were over $500 last
                 │         quarter?" -- the exact number now comes from
                 │         the policy doc, not a guess)
                 │
                 ├─ 3. answer_quantitative_question(sub_question)
                 │     → same pipeline as the simple quantitative case
                 │     (1-2 Gemini calls internally)
                 │
                 ├─ 4. _synthesize_and_check()
                 │     Gemini call: combine both results into one answer
                 │     AND self-assess completeness in the same call
                 │     → {"answer": "...", "complete": bool, "gap": "..."}
                 │
                 │     Deterministic override: if the quantitative results
                 │     are all-null (e.g. a WHERE clause matched zero rows),
                 │     force complete=False regardless of Gemini's self-report
                 │     -- this catches cases the LLM can miss on its own
                 │     (see README, "Lessons Learned" #1 and the manager
                 │     agent test suite for the specific bug this fixes).
                 │
                 └─ 5. IF NOT complete: one clarifying follow-up
                       ├─ re-run the quantitative sub-question with the
                       │   gap as added context
                       └─ re-run _synthesize_and_check() once more
                          (not looped -- exactly one retry)
                 │
                 └─→ CLI displays final answer + sources + SQL +
                     a note if a follow-up was used
```

Total Gemini calls: 4-5 typically, 6-7 if a follow-up is triggered. This is why `_synthesize_and_check` merges what was originally two separate calls (synthesis + completeness check) into one -- see README "Lessons Learned" #4 for why this mattered in practice (free-tier quota pressure).

## Data Flow Summary

```
                     ┌──────────────────────┐
                     │   data/docs/*.md      │  (policy documents)
                     └──────────┬───────────┘
                                │ chunked by ## section headers
                                ▼
                     ┌──────────────────────┐
                     │  Sentence Transformers │  (local embeddings)
                     └──────────┬───────────┘
                                ▼
                     ┌──────────────────────┐
                     │   chroma_db/          │  (persistent vector store)
                     └──────────────────────┘
                                ▲
                                │ semantic search (top-3)
                     ┌──────────┴───────────┐
                     │  Qualitative Agent    │
                     └──────────────────────┘

                     ┌──────────────────────┐
                     │ data/enterprise.db    │  (revenue, customers,
                     │  (SQLite)             │   tickets, expenses)
                     └──────────┬───────────┘
                                │ schema introspection
                                │ (+ sample values, + category lists)
                                ▼
                     ┌──────────────────────┐
                     │  Quantitative Agent   │  → validate_sql() → execute
                     └──────────────────────┘
```

## Sample Query Routing Reference

| Question | Classification | Why |
|---|---|---|
| "What is our company's security policy?" | qualitative | Answerable entirely from policy docs |
| "Explain the code review process" | qualitative | Answerable entirely from policy docs |
| "What's our customer churn rate?" | quantitative | Answerable entirely from database numbers |
| "Compare Q4 performance across regions" | quantitative | Answerable entirely from database numbers |
| "What's our policy on expense approvals, and how many expense requests last quarter would have required manager sign-off under that policy?" | complex | Needs the $500 threshold (policy) AND a count (database) -- deliberately phrased to test that classification reasons about content, not the word "expenses" alone |
| "Based on our documented code review process, are our current code review turnaround times meeting the standard we've committed to?" | complex | Needs the 48-hour standard (policy) AND turnaround data filtered to `code_review` tickets specifically (database) |