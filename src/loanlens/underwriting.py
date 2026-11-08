"""Human-in-the-loop: an underwriter signs off every referred case. These are the graph nodes;
cases.py has the functions the API uses to read a pending review and resume the case.

    refer_to_underwriter -> underwriter_review    interrupt(): the graph pauses and saves its state
    route_after_review:  invalid decision -> underwriter_review (ask again)
                         approve -> approve_in_principle;  decline -> underwriter_decline
"""

from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from loanlens.state import PipelineState

ALLOWED_DECISIONS = ("approve", "decline")


def review_payload(state: PipelineState) -> dict[str, Any]:
    """What the underwriter sees: everything needed to decide, nothing more (and JSON-serialisable).

    The application is already redacted.
    """
    return {
        "application": state["application"],
        "risk_flags": state["risk_flags"],
        "error": state.get("review_error"),
        "applicant_updates": state.get("held_messages", []),
        "specialist_reports": state.get("specialist_reports", {}),
    }


def validate_decision(decision: Any) -> str | None:
    """Return None if the decision is valid, otherwise a message saying what's wrong.

    Valid means a dict with "decision" in ALLOWED_DECISIONS, a non-empty "reason" and a non-empty
    "underwriter_id": an audit trail needs a name and a why.
    """
    if not isinstance(decision, dict):
        return "Decision must be an object with decision, reason and underwriter_id."
    if decision.get("decision") not in ALLOWED_DECISIONS:
        return f"decision must be one of {ALLOWED_DECISIONS}."
    if not str(decision.get("reason", "")).strip():
        return "A reason is required for the audit trail."
    if not str(decision.get("underwriter_id", "")).strip():
        return "underwriter_id is required for the audit trail."
    return None


def underwriter_review(state: PipelineState) -> dict:
    """Node: pause for the underwriter's decision, then validate it.

    interrupt() saves the state and stops the run. When the case is resumed, this node runs again
    from its first line and interrupt() returns the decision, so nothing before it may have side effects.
    """
    decision = interrupt(review_payload(state))
    error = validate_decision(decision)
    if error:
        return {"review_error": error}
    return {"underwriter_decision": decision, "review_error": None}


def route_after_review(
    state: PipelineState,
) -> Literal["underwriter_review", "approve_in_principle", "underwriter_decline"]:
    """Invalid decision -> ask again; approve -> approve_in_principle; decline -> underwriter_decline."""
    if state.get("review_error"):
        return "underwriter_review"
    if state["underwriter_decision"]["decision"] == "approve":
        return "approve_in_principle"
    return "underwriter_decline"


def underwriter_decline(state: PipelineState) -> dict:
    return {
        "outcome": "declined_by_underwriter",
        "messages": [AIMessage("After review by our underwriting team, we're unable to offer this loan.")],
    }
