"""Screening: policy checks that run in parallel, then combine.

Every check writes to risk_flags in the same super-step, so risk_flags has an append reducer.
check_sanctions runs for every complete application, and a match is a flag, so it always reaches a
human. build_graph compiles the checks as a standalone graph for tests.
"""

from typing import Annotated
from operator import add

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from loanlens.compliance import is_sanctioned

# LoanLens policy (illustrative): every rule traces back to here
MIN_AGE = 18
MAX_LOAN_TO_INCOME = 0.5  # flag if loan is MORE than 50% of annual income


class ScreeningState(MessagesState, total=False):
    # `messages` is inherited from MessagesState: list of messages, reducer = add_messages
    application: dict
    risk_flags: Annotated[
        list[str], add
    ]  # parallel checks append here
    decision_hint: str  # "clear" | "refer"


def check_affordability(state: ScreeningState) -> dict:
    """Flag high_loan_to_income if the loan is more than MAX_LOAN_TO_INCOME of annual income."""
    loan_amount = state["application"]["loan_amount"]
    annual_income = state["application"]["annual_income"]
    risk_flags = []
    if loan_amount / annual_income > MAX_LOAN_TO_INCOME:
        risk_flags.append("high_loan_to_income")
    return {"risk_flags": risk_flags}


def check_age(state: ScreeningState) -> dict:
    """Flag under_18 if the applicant is younger than MIN_AGE."""
    age = state["application"]["age"]
    if not age:
        return {"risk_flags": []}
    return {"risk_flags": ["under_18"] if age < MIN_AGE else []}


def check_employment(state: ScreeningState) -> dict:
    """Flag unemployed and self-employed applicants."""
    employment_status = state["application"].get("employment_status")
    risk_flags = []
    if employment_status == "unemployed":
        risk_flags.append("no_employment_income")
    elif employment_status == "self_employed":
        risk_flags.append("self_employed_needs_review")
    return {"risk_flags": risk_flags}


def combine(state: ScreeningState) -> dict:
    """Turn the flags into a decision hint and one summary message."""
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
    """Standalone screening graph: every check in parallel, then combine."""
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
