"""Intake: validate an extracted application.

validate_application is part of the pipeline. compute_ratios, summarise and build_graph form a
small standalone intake graph used in tests.
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
    """Return missing_fields (in REQUIRED_FIELDS order) and is_complete.

    A field is missing if its key is absent or its value is None.
    """
    app = state["application"]
    missing = [field for field in REQUIRED_FIELDS if app.get(field) is None]
    return {"missing_fields": missing, "is_complete": not missing}


def compute_ratios(state: IntakeState) -> dict:
    """Loan-to-income ratio to 2 dp, or None if a value is missing or income is 0."""
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
    """Status for the standalone intake graph: ready_for_review or incomplete."""
    return {"status": "ready_for_review" if state["is_complete"] else "incomplete"}


def build_graph():
    """Standalone intake graph: validate_application -> compute_ratios -> summarise."""
    builder = StateGraph(IntakeState)
    builder.add_node(validate_application)
    builder.add_node(compute_ratios)
    builder.add_node(summarise)
    builder.add_edge(START, "validate_application")
    builder.add_edge("validate_application", "compute_ratios")
    builder.add_edge("compute_ratios", "summarise")
    builder.add_edge("summarise", END)

    return builder.compile()
