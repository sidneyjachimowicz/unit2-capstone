"""
manager_agent_graph.py

LangGraph refactor of the Manager Agent's routing logic (Silver stretch
goal). Replaces the custom if/else routing in manager_agent.handle_question
with an explicit StateGraph: classify -> route -> (qualitative | quantitative
| decompose -> synthesize+check -> [followup]) -> END.

Reuses every underlying function from manager_agent.py (classification,
decomposition, synthesis, agent calls) unchanged -- only the control flow
is restructured into a graph. This keeps the well-tested business logic
intact while satisfying the "refactor routing logic using LangGraph"
requirement.
"""

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from src.manager_agent import (
    classify_question,
    _decompose_complex_question,
    _synthesize_and_check,
)
from src.qualitative_agent import answer_qualitative_question
from src.quantitative_agent import answer_quantitative_question


class ManagerState(TypedDict, total=False):
    question: str
    classification: dict
    qual_result: dict
    quant_result: dict
    quant_subquestion: str
    synthesis: dict
    answer: object
    sources: list
    sql: Optional[str]
    followup_used: bool


# --- Nodes --------------------------------------------------------------

def classify_node(state: ManagerState) -> ManagerState:
    classification = classify_question(state["question"])
    return {"classification": classification}


def qualitative_node(state: ManagerState) -> ManagerState:
    result = answer_qualitative_question(state["question"])
    return {
        "answer": result.get("answer"),
        "sources": result.get("sources", []),
    }


def quantitative_node(state: ManagerState) -> ManagerState:
    result = answer_quantitative_question(state["question"])
    return {
        "answer": result.get("results") if result.get("success") else result.get("error"),
        "sql": result.get("sql"),
    }


def decompose_node(state: ManagerState) -> ManagerState:
    qual_result, quant_result, quant_subquestion = _decompose_complex_question(state["question"])
    return {
        "qual_result": qual_result,
        "quant_result": quant_result,
        "quant_subquestion": quant_subquestion,
    }


def synthesize_node(state: ManagerState) -> ManagerState:
    synthesis = _synthesize_and_check(state["question"], state["qual_result"], state["quant_result"])
    return {
        "synthesis": synthesis,
        "answer": synthesis["answer"],
        "sources": state["qual_result"].get("sources", []),
        "sql": state["quant_result"].get("sql"),
        "followup_used": state.get("followup_used", False),
    }


def followup_node(state: ManagerState) -> ManagerState:
    """
    Exactly one clarifying follow-up: re-run the quantitative sub-question
    with the completeness gap as added context, then re-synthesize.
    """
    gap = state["synthesis"].get("gap", "")
    quant_result_2 = answer_quantitative_question(
        f"{state['quant_subquestion']}\n(Additional context: {gap})"
    )
    synthesis_2 = _synthesize_and_check(state["question"], state["qual_result"], quant_result_2)
    return {
        "quant_result": quant_result_2,
        "synthesis": synthesis_2,
        "answer": synthesis_2["answer"],
        "sql": quant_result_2.get("sql"),
        "followup_used": True,
    }


# --- Conditional routing --------------------------------------------------

def route_by_classification(state: ManagerState) -> str:
    label = state["classification"]["classification"]
    if label == "qualitative":
        return "qualitative"
    if label == "quantitative":
        return "quantitative"
    return "decompose"  # complex, or an unrecognized label (fail toward the safer complex path)


def route_by_completeness(state: ManagerState) -> str:
    if state["synthesis"].get("complete", True):
        return "done"
    return "followup"


# --- Graph construction ---------------------------------------------------

def build_graph():
    graph = StateGraph(ManagerState)

    graph.add_node("classify", classify_node)
    graph.add_node("qualitative", qualitative_node)
    graph.add_node("quantitative", quantitative_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("followup", followup_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_by_classification,
        {
            "qualitative": "qualitative",
            "quantitative": "quantitative",
            "decompose": "decompose",
        },
    )

    graph.add_edge("qualitative", END)
    graph.add_edge("quantitative", END)
    graph.add_edge("decompose", "synthesize")

    graph.add_conditional_edges(
        "synthesize",
        route_by_completeness,
        {
            "done": END,
            "followup": "followup",
        },
    )

    graph.add_edge("followup", END)

    return graph.compile()


_compiled_graph = None


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def handle_question(question: str) -> dict:
    """
    Drop-in replacement for manager_agent.handle_question, same return
    shape, but routed through the LangGraph StateGraph above instead of
    a hand-written if/else chain.
    """
    graph = _get_graph()
    final_state = graph.invoke({"question": question})

    return {
        "question": question,
        "classification": final_state.get("classification"),
        "answer": final_state.get("answer"),
        "sources": final_state.get("sources", []),
        "sql": final_state.get("sql"),
        "followup_used": final_state.get("followup_used", False),
    }


if __name__ == "__main__":
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