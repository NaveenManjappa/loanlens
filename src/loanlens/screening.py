"""Lesson 2: LoanLens screening. Parallel checks + reducers.

             ┌─> check_affordability ─┐
    START ───┼─> check_age ───────────┼──> combine -> END
             └─> check_employment ────┘

The three checks run IN PARALLEL (same super-step) and each one writes
to the SAME key: `risk_flags`. Make that work.

YOUR JOB: make  `uv run pytest tests/test_lesson02_screening.py -v`  go green.
Hint: one of the failing tests will only pass when you change the STATE,
not a node. Read the error message carefully; it tells you what to do.
"""

from typing import Annotated  # noqa: F401  (you'll need this)
from operator import add  # noqa: F401  (and probably this)

from langchain_core.messages import AIMessage  # noqa: F401
from langgraph.graph import END, START, MessagesState, StateGraph

from loanlens.compliance import is_sanctioned

# LoanLens policy v0 (made up for the course — every rule traces back to here)
MIN_AGE = 18
MAX_LOAN_TO_INCOME = 0.5  # flag if loan is MORE than 50% of annual income


class ScreeningState(MessagesState, total=False):
    # `messages` is inherited from MessagesState: list of messages, reducer = add_messages
    application: dict
    risk_flags: Annotated[
        list[str], add
    ]  # TODO(Lesson 2): three parallel nodes write here. What does this key need?
    decision_hint: str  # "clear" | "refer"


def check_affordability(state: ScreeningState) -> dict:
    """Return {"risk_flags": ["high_loan_to_income"]} if loan_amount / annual_income
    is MORE than MAX_LOAN_TO_INCOME, otherwise {"risk_flags": []}.
    (Assume both values are present. Validation happened upstream.)"""
    loan_amount = state["application"]["loan_amount"]
    annual_income = state["application"]["annual_income"]
    risk_flags = []
    if loan_amount / annual_income > MAX_LOAN_TO_INCOME:
        risk_flags.append("high_loan_to_income")
    return {"risk_flags": risk_flags}


def check_age(state: ScreeningState) -> dict:
    """Return {"risk_flags": ["under_18"]} if age < MIN_AGE, otherwise {"risk_flags": []}."""
    age = state["application"]["age"]
    if not age:
        return {"risk_flags": []}
    return {"risk_flags": ["under_18"] if age < MIN_AGE else []}


def check_employment(state: ScreeningState) -> dict:
    """employment_status -> flag:
        "unemployed"     -> ["no_employment_income"]
        "self_employed"  -> ["self_employed_needs_review"]
        anything else    -> []
    Return {"risk_flags": [...]}."""
    employment_status = state["application"].get("employment_status")
    risk_flags = []
    if employment_status == "unemployed":
        risk_flags.append("no_employment_income")
    elif employment_status == "self_employed":
        risk_flags.append("self_employed_needs_review")
    return {"risk_flags": risk_flags}


def combine(state: ScreeningState) -> dict:
    """Runs once, AFTER all three checks. Return:
        "decision_hint": "refer" if there are any risk_flags, else "clear"
        "messages": [AIMessage(...)]  - ONE new message whose text contains the
                    decision_hint and the flag names (sorted), e.g.
                    "Screening: refer. Flags: self_employed_needs_review, under_18"
                    (use "none" when there are no flags)
    Return only the NEW message. The reducer takes care of appending it."""
    risk_flags = state["risk_flags"]
    decision_hint = "refer" if risk_flags else "clear"

    return {
        "decision_hint": decision_hint,
        "messages": [
            AIMessage(
                f"Screening: {decision_hint}. Flags: {', '.join(sorted(risk_flags)) if risk_flags else 'none'}"
            )
        ],
    }
def check_sanctions(state: ScreeningState) -> dict:
    applicant_name = state["application"]["applicant_name"]
    return {"risk_flags": ["sanctions_match"] if is_sanctioned(applicant_name) else []}

def build_graph():
    """START fans out to all three checks; all three fan in to combine; combine -> END.
    That's 3 edges out of START, 3 edges into combine, 1 edge to END. Compile it."""
    builder = StateGraph(ScreeningState)
    builder.add_node(check_affordability)
    builder.add_node(check_age)
    builder.add_node(check_employment)
    builder.add_node(check_sanctions)
    builder.add_node(combine)
    builder.add_edge(START, "check_affordability")
    builder.add_edge(START, "check_age")
    builder.add_edge(START, "check_employment")
    builder.add_edge(START, "check_sanctions")
    builder.add_edge("check_sanctions", "combine")
    builder.add_edge("check_affordability", "combine")
    builder.add_edge("check_age", "combine")
    builder.add_edge("check_employment", "combine")
    builder.add_edge("combine", END)
    return builder.compile()
