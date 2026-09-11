# Tokenomics Test Run

Recorded run of the Enterprise RAG System via the CLI (`python -m src.cli`), demonstrating token usage and cost tracking across all three agents per the project's tokenomics requirement.

## Questions Asked

1. "What is our company's security policy?" (qualitative)
2. "Explain the code review process" (qualitative)
3. "What's our customer churn rate?" (quantitative)
4. "Compare Q4 performance across regions" (quantitative)
5. "How do we handle customer complaints?" (qualitative)
6. "What's our policy on expense approvals?" (qualitative)
7. "What's our average code review turnaround time?" (quantitative)
8. "How many expense requests were over $500 last quarter?" (quantitative)

Both of the project's required "harder queries" were also confirmed working live in separate runs during development (see below), bringing the total number of distinct live-tested questions across the full session to 10, though not all landed inside a single uninterrupted CLI session due to the free-tier quota constraint documented below.

**Harder query 1 (red-herring, tested separately):**
"What's our policy on expense approvals, and how many expense requests last quarter would have required manager sign-off under that policy?" → Correctly classified as `complex` (not misled by quantitative-sounding language), correctly decomposed into a policy lookup ($500 threshold) and a database count, correctly landed on **40 expense requests**.

**Harder query 2 (multi-step, tested separately):**
"Based on our documented code review process, are our current code review turnaround times meeting the standard we've committed to?" → Correctly classified as `complex`, correctly filtered the database query to `type = 'code_review'` specifically, correctly compared against the 48-hour standard from policy, landed on **78.79%** meeting the standard (matching the ~79% baked into the seed data).

## Results Summary

```
==================================================
TOKENOMICS SUMMARY
==================================================
Total calls: 17
Total input tokens:  5127
Total output tokens: 1448
Total cost: $0.000205

Per-agent breakdown:
  QualitativeAgent: 4 calls, in=1240, out=996, cost=$0.000098
  ManagerAgent: 9 calls, in=2057, out=334, cost=$0.000064
  QuantitativeAgent: 4 calls, in=1830, out=118, cost=$0.000043

Highest token consumer: ManagerAgent
==================================================
```

**Highest token consumer: Manager Agent** — expected, since it performs classification for every single question (a call qualitative/quantitative-only questions don't otherwise need) plus decomposition and synthesis for any complex questions.

**Total cost for this entire run: $0.000205** — a fraction of a cent for 17 real Gemini API calls.

## Known Constraint: Free-Tier Daily Quota

This project's Google Gemini free tier caps `gemini-3.6-flash` at **20 requests/day per Google Cloud project**. During development and testing, this session's project reached that daily cap after the 8 questions above (each qualitative/quantitative question costs ~1-2 calls; the two harder complex queries each cost 4-5 calls on their own when tested separately), which is why the 9th and 10th questions in a single continuous CLI session returned `429 RESOURCE_EXHAUSTED` errors rather than answers.

This is a real, externally-imposed constraint rather than a bug in the system: the code's retry logic (`src/gemini_utils.py`) correctly distinguishes this from a transient error and fails fast rather than wasting time on a retry that cannot succeed until the daily quota resets. Given the measured cost above (~$0.00002 per call on average), enabling billing on the Google Cloud project removes this cap entirely for a cost of well under $1 for extensive testing.

10+ distinct questions, covering every required query type including both harder queries, were successfully answered live across this development session; they were not all captured in a single unbroken 20-call window due to this quota, which is documented here rather than glossed over.