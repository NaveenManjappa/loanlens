"""Lesson 1 — the LoanLens intake pipeline. Your first StateGraph (no LLM yet).

    START -> validate_application -> compute_ratios -> summarise -> END

YOUR JOB: replace every `raise NotImplementedError` so that
    uv run pytest tests/test_lesson01_intake.py
goes green. Read the test file first — it is the spec.

Rules of the game (from the LangGraph Graph API docs):
  * A node is a plain function: it receives the WHOLE state, and returns a
    dict containing ONLY the keys it wants to change (a "partial update").
  * Never mutate `state` in place. Return a new dict instead.
  * Edges say which node runs next. START and END are special built-in nodes.
  * You must .compile() the builder before you can .invoke() it.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

REQUIRED_FIELDS = [
    "applicant_name",
    "age",
    "annual_income",
    "loan_amount",
    "employment_status",
]


class IntakeState(TypedDict, total=False):
    # Input — set by whoever calls graph.invoke({...})
    application: dict
    # Written by validate_application
    missing_fields: list[str]
    is_complete: bool
    # Written by compute_ratios
    loan_to_income: float | None
    # Written by summarise
    status: str  # "ready_for_review" | "incomplete"


def validate_application(state: IntakeState) -> dict:
    """Return {"missing_fields": [...], "is_complete": bool}.

    A field counts as missing if the key is absent OR its value is None.
    Keep missing_fields in the same order as REQUIRED_FIELDS.
    """
    # resturnValue = {"missing_fields": [], "is_complete": True}
    # for key in REQUIRED_FIELDS:
    #     if key not in state["application"]:
    #         resturnValue["missing_fields"].append(key)
    #         resturnValue["is_complete"] = False
    #     elif state["application"][key] is None:
    #         resturnValue["missing_fields"].append(key)
    #         resturnValue["is_complete"] = False
    # return resturnValue
    app = state["application"]
    missing = [field for field in REQUIRED_FIELDS if app.get(field) is None]
    return {"missing_fields": missing, "is_complete": not missing}


def compute_ratios(state: IntakeState) -> dict:
    """Return {"loan_to_income": loan_amount / annual_income, rounded to 2 dp}.

    If either value is missing (or income is 0), return {"loan_to_income": None}
    instead of crashing.  (Notice how ugly this defensive check feels —
    Lesson 4 replaces it with routing.)
    """
    if (
        not "loan_amount" in state["application"]
        or not "annual_income" in state["application"]
    ):
        return {"loan_to_income": None}

    loan_amount = state["application"]["loan_amount"]
    annual_income = state["application"]["annual_income"]

    if loan_amount is None or annual_income is None or annual_income == 0:
        return {"loan_to_income": None}

    return {
        "loan_to_income": round(
            state["application"]["loan_amount"] / state["application"]["annual_income"],
            2,
        )
    }


def summarise(state: IntakeState) -> dict:
    """Return {"status": "ready_for_review"} if the application is complete,
    otherwise {"status": "incomplete"}."""
    # if state["is_complete"]:
    #     return {"status": "ready_for_review"}
    # else:
    #     return {"status": "incomplete"}
    return {"status": "ready_for_review" if state["is_complete"] else "incomplete"}


def build_graph():
    """Wire the three nodes in a straight line and return the COMPILED graph.

    Steps: create StateGraph(IntakeState) -> add_node x3 -> add_edge x4 -> compile().
    Use the function names as node names.
    """
    builder = StateGraph(IntakeState)
    builder.add_node(validate_application)
    builder.add_node(compute_ratios)
    builder.add_node(summarise)
    builder.add_edge(START, "validate_application")
    builder.add_edge("validate_application", "compute_ratios")
    builder.add_edge("compute_ratios", "summarise")
    builder.add_edge("summarise", END)

    return builder.compile()
