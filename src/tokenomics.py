"""
tokenomics.py

Tracks token usage and cost across every Gemini API call made by any agent.
In-memory only: call log_usage() after every Gemini call, then print_summary()
at the end of a session (e.g. after a 10-query test run) to see totals.
"""

# Pricing as of this project's build date — verify against current Gemini
# pricing before relying on these numbers for real budgeting.
COST_PER_1K_INPUT = 0.00001875
COST_PER_1K_OUTPUT = 0.000075

# In-memory log of every call made this session.
_usage_log = []


def log_usage(agent_name: str, input_tokens: int, output_tokens: int) -> dict:
    """
    Record one Gemini API call's token usage and cost.
    Call this immediately after every generate_content() call,
    in every agent, no exceptions.
    """
    cost = (input_tokens / 1000 * COST_PER_1K_INPUT) + (
        output_tokens / 1000 * COST_PER_1K_OUTPUT
    )
    entry = {
        "agent": agent_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }
    _usage_log.append(entry)
    print(
        f"[{agent_name}] tokens: in={input_tokens} out={output_tokens} "
        f"cost=${cost:.6f}"
    )
    return entry


def print_summary():
    """
    Print a summary of every logged call this session: total cost,
    total tokens, and a per-agent breakdown showing which agent
    consumed the most tokens.
    """
    if not _usage_log:
        print("No usage logged yet.")
        return

    total_cost = sum(e["cost"] for e in _usage_log)
    total_in = sum(e["input_tokens"] for e in _usage_log)
    total_out = sum(e["output_tokens"] for e in _usage_log)

    per_agent = {}
    for e in _usage_log:
        agent = e["agent"]
        if agent not in per_agent:
            per_agent[agent] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}
        per_agent[agent]["input_tokens"] += e["input_tokens"]
        per_agent[agent]["output_tokens"] += e["output_tokens"]
        per_agent[agent]["cost"] += e["cost"]
        per_agent[agent]["calls"] += 1

    print("\n" + "=" * 50)
    print("TOKENOMICS SUMMARY")
    print("=" * 50)
    print(f"Total calls: {len(_usage_log)}")
    print(f"Total input tokens:  {total_in}")
    print(f"Total output tokens: {total_out}")
    print(f"Total cost: ${total_cost:.6f}")
    print("\nPer-agent breakdown:")
    for agent, stats in sorted(per_agent.items(), key=lambda x: -x[1]["cost"]):
        print(
            f"  {agent}: {stats['calls']} calls, "
            f"in={stats['input_tokens']}, out={stats['output_tokens']}, "
            f"cost=${stats['cost']:.6f}"
        )

    top_agent = max(per_agent.items(), key=lambda x: x[1]["input_tokens"] + x[1]["output_tokens"])
    print(f"\nHighest token consumer: {top_agent[0]}")
    print("=" * 50)


def get_log():
    """Return the raw in-memory usage log, e.g. for tests."""
    return _usage_log


def reset_log():
    """Clear the log. Useful between test runs."""
    _usage_log.clear()