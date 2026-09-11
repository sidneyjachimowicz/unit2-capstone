"""
manager_agent.py

The Manager Agent classifies incoming questions (qualitative, quantitative,
or complex) by reasoning about their content -- not by keyword matching --
then routes to the appropriate agent(s), synthesizes multi-agent answers,
and checks whether the final answer actually addresses the original
question, issuing one clarifying follow-up if not.
"""

import os
import json
import re
from dotenv import load_dotenv
from google import genai

from src.qualitative_agent import answer_qualitative_question
from src.quantitative_agent import answer_quantitative_question
from src.tokenomics import log_usage
from src.gemini_utils import generate_with_retry

load_dotenv()

MODEL_NAME = "gemini-3.6-flash"
AGENT_NAME = "ManagerAgent"

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment.")
        _client = genai.Client(api_key=api_key)
    return _client


def _call_gemini(prompt: str) -> str:
    """Shared helper: call Gemini, log tokens, return raw text."""
    client = _get_client()
    response = generate_with_retry(client, MODEL_NAME, prompt)
    usage = response.usage_metadata
    log_usage(
        AGENT_NAME,
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
    )
    return response.text.strip()


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of Gemini's response, tolerating markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def classify_question(question: str) -> dict:
    """
    Ask Gemini to classify the question by reasoning about its content and
    what data would actually be needed to answer it -- explicitly NOT by
    matching keywords. Returns {"classification": ..., "reasoning": ...}.
    """
    prompt = f"""You are the routing brain of a RAG system with two specialist agents:

- A QUALITATIVE agent that searches company policy documents (semantic search).
- A QUANTITATIVE agent that queries a SQL database of business metrics
  (revenue, customers, tickets, expenses).

Classify the following question as one of:
- "qualitative": answerable purely from policy documents
- "quantitative": answerable purely from database numbers
- "complex": requires BOTH a policy fact AND a database number to answer fully

IMPORTANT: Reason about what information is actually needed to answer the
question completely -- do not classify based on surface keywords alone.
A question can mention "expenses" or sound numeric while still requiring a
policy lookup too, or vice versa. Think about what a complete, correct
answer would actually require.

Question: {question}

Respond with ONLY a JSON object, no markdown fences, in this exact format:
{{"classification": "qualitative" | "quantitative" | "complex", "reasoning": "one sentence explaining why"}}
"""
    raw = _call_gemini(prompt)
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        # Fallback: if Gemini didn't return clean JSON, default to complex
        # so we don't silently under-answer a question we couldn't parse.
        return {"classification": "complex", "reasoning": f"Could not parse classifier output: {raw[:200]}"}


def _decompose_complex_question(question: str) -> str:
    """
    For a complex question, first get the relevant policy fact, then use
    that fact to build a precise quantitative sub-question (since the two
    harder queries in this project both depend on a policy-defined
    threshold/standard before the numeric query makes sense).
    """
    qual_result = answer_qualitative_question(question)
    qual_answer = qual_result.get("answer") or "No relevant policy found."

    decompose_prompt = f"""Original question: {question}

Relevant policy information found: {qual_answer}

Based on the policy information above, write ONE precise, self-contained
natural language question that a database agent could answer using SQL,
that would provide the missing numeric piece needed to fully answer the
original question. Use any specific numbers/thresholds from the policy
information directly in your question. Respond with ONLY the question,
nothing else.
"""
    quant_subquestion = _call_gemini(decompose_prompt)
    quant_result = answer_quantitative_question(quant_subquestion)

    return qual_result, quant_result, quant_subquestion


def _synthesize_and_check(question: str, qual_result: dict, quant_result: dict) -> dict:
    """
    Combined synthesis + completeness self-check in ONE Gemini call
    (instead of two) to conserve API quota. Returns
    {"answer": str, "complete": bool, "gap": str}.
    """
    qual_part = qual_result.get("answer") or "No policy information available."
    quant_part = (
        f"SQL query used: {quant_result.get('sql')}\nResults: {quant_result.get('results')}"
        if quant_result.get("success")
        else f"Could not retrieve data: {quant_result.get('error')}"
    )

    prompt = f"""Original question: {question}

Policy information:
{qual_part}

Database findings:
{quant_part}

Do two things:
1. Write one clear, complete answer to the original question that combines
   both pieces of information. Be specific and cite the numbers found.
2. Self-assess: does your answer fully and directly address every part of
   the original question, with no missing pieces?

Respond with ONLY a JSON object, no markdown fences, in this exact format:
{{"answer": "your full answer here", "complete": true or false, "gap": "if incomplete, one sentence on what's missing, else empty string"}}
"""
    raw = _call_gemini(prompt)
    try:
        return _extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        # fail open: treat the raw text as the answer and assume complete
        # rather than looping or crashing on a parse hiccup
        return {"answer": raw, "complete": True, "gap": ""}


def handle_question(question: str) -> dict:
    """
    Main entry point. Classifies, routes, and (for complex questions)
    synthesizes + checks completeness with one clarifying follow-up if
    the first pass falls short.
    """
    classification = classify_question(question)
    label = classification["classification"]

    if label == "qualitative":
        result = answer_qualitative_question(question)
        return {
            "question": question,
            "classification": classification,
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
        }

    if label == "quantitative":
        result = answer_quantitative_question(question)
        return {
            "question": question,
            "classification": classification,
            "answer": result.get("results") if result.get("success") else result.get("error"),
            "sql": result.get("sql"),
        }

    # complex
    qual_result, quant_result, quant_subquestion = _decompose_complex_question(question)
    synthesis = _synthesize_and_check(question, qual_result, quant_result)
    answer = synthesis["answer"]
    followup_used = False

    if not synthesis.get("complete", True):
        followup_used = True
        # One clarifying follow-up: re-run the quantitative sub-question
        # with the gap as extra context, then re-synthesize (still just
        # one more combined call, not two).
        quant_result_2 = answer_quantitative_question(
            f"{quant_subquestion}\n(Additional context: {synthesis.get('gap')})"
        )
        synthesis = _synthesize_and_check(question, qual_result, quant_result_2)
        answer = synthesis["answer"]
        quant_result = quant_result_2

    return {
        "question": question,
        "classification": classification,
        "answer": answer,
        "sources": qual_result.get("sources", []),
        "sql": quant_result.get("sql"),
        "followup_used": followup_used,
    }


if __name__ == "__main__":
    # Quick smoke tests: one of each type, including the red-herring query.
    tests = [
        "What is our company's security policy?",
        "What's our customer churn rate?",
        "What's our policy on expense approvals, and how many expense requests last quarter would have required manager sign-off under that policy?",
    ]
    for q in tests:
        print("\n" + "=" * 70)
        print("Q:", q)
        result = handle_question(q)
        print("Classification:", result["classification"])
        print("Answer:", result["answer"])