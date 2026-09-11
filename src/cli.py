"""
cli.py

Command-line interface for the Manager Agent. Accepts user queries in an
interactive loop, routes them through classification, and prints answers
with source attribution / SQL where relevant. Type 'exit' or 'quit' to
end the session and see a tokenomics summary for everything asked.
"""

import sys
from src.manager_agent import handle_question
from src.tokenomics import print_summary

BANNER = """
==============================================================
  Enterprise RAG System — Manager Agent CLI
  Ask a qualitative, quantitative, or complex question below.
  Type 'exit' or 'quit' to end the session.
==============================================================
"""

EXIT_COMMANDS = {"exit", "quit", "q"}


def format_result(result: dict) -> str:
    lines = []
    classification = result.get("classification", {})
    label = classification.get("classification", "unknown")
    reasoning = classification.get("reasoning", "")

    lines.append(f"\n[Classified as: {label}]")
    if reasoning:
        lines.append(f"  Reasoning: {reasoning}")

    lines.append(f"\nAnswer:\n{result.get('answer')}")

    sources = result.get("sources")
    if sources:
        lines.append("\nSources:")
        for s in sources:
            lines.append(f"  - {s.get('doc_name')} ({s.get('section')})")

    sql = result.get("sql")
    if sql:
        lines.append(f"\nSQL used:\n  {sql}")

    if result.get("followup_used"):
        lines.append("\n[Note: a clarifying follow-up query was used to complete this answer.]")

    return "\n".join(lines)


def main():
    print(BANNER)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nExiting.")
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            print("\nExiting.")
            break

        try:
            result = handle_question(question)
            print(format_result(result))
        except Exception as e:
            print(f"\n[Error] Something went wrong answering that question: {e}")
            print("Try again, or type 'exit' to quit.")

    print("\n" + "=" * 62)
    print_summary()


if __name__ == "__main__":
    sys.exit(main())