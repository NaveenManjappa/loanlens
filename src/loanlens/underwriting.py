"""Human-in-the-loop: an underwriter signs off every referred case.

    refer_to_underwriter -> underwriter_review    interrupt(): the graph pauses and saves its state
    route_after_review:  invalid decision -> underwriter_review (ask again)
                         approve -> approve_in_principle;  decline -> underwriter_decline

interrupt() is called exactly once per node run; a validation error is stored in state and the graph
loops back. While a review is pending, applicant messages are held for the underwriter
(hold_for_underwriter) instead of starting a new run (ADR 0001).
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt

from loanlens.cases import thread_config
from loanlens.pii import redact_at_boundary

ALLOWED_DECISIONS = ("approve", "decline")


def review_payload(state) -> dict:
    """What the underwriter sees: everything needed to decide, nothing more.

    The application is already redacted. The payload is JSON-serialisable.
    """
    return {
        "application": state["application"],
        "risk_flags": state["risk_flags"],
        "error": state.get("review_error"),
        "applicant_updates": state.get("held_messages", []),
        "specialist_reports": state.get("specialist_reports", {}),
    }


def validate_decision(decision) -> str | None:
    """None if the decision is valid, otherwise a message explaining what's wrong.
    Valid = a dict with "decision" in ALLOWED_DECISIONS, a non-empty "reason",
    and a non-empty "underwriter_id" (an audit trail needs a NAME and a WHY)."""
    if not isinstance(decision, dict):
        return "Decision must be an object with decision, reason and underwriter_id."
    if decision.get("decision") not in ALLOWED_DECISIONS:
        return f"decision must be one of {ALLOWED_DECISIONS}."
    if not str(decision.get("reason", "")).strip():
        return "A reason is required for the audit trail."
    if not str(decision.get("underwriter_id", "")).strip():
        return "underwriter_id is required for the audit trail."
    return None


def underwriter_review(state) -> dict:
    """Node: pause for the underwriter's decision, then validate it.

    On resume this node re-runs from its first line, so everything before interrupt() must be
    side-effect free.
    """
    decision = interrupt(review_payload(state))
    error = validate_decision(decision)
    if error:
        return {"review_error": error}
    return {"underwriter_decision": decision, "review_error": None}


def route_after_review(
    state,
) -> Literal["underwriter_review", "approve_in_principle", "underwriter_decline"]:
    """Invalid decision -> ask again; approve -> approve_in_principle; otherwise underwriter_decline."""
    if state.get("review_error"):
        return "underwriter_review"
    if state.get("underwriter_decision")["decision"] == "approve":
        return "approve_in_principle"
    return "underwriter_decline"


def underwriter_decline(state) -> dict:
    return {
        "outcome": "declined_by_underwriter",
        "messages": [
            AIMessage(
                "After review by our underwriting team, we're unable to offer this loan."
            )
        ],
    }


# ── Helpers for the underwriter's app (no graph changes) ───────────────────


def awaiting_underwriter(graph, thread_id: str) -> dict | None:
    """The review payload if the case is paused for an underwriter, otherwise None.

    Also covers a paused case whose interrupt marker was cleared by update_state: its .next still
    points at underwriter_review.
    """
    state = graph.get_state(thread_config(thread_id))
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    if state.next == (
        "underwriter_review",
    ):  # paused, but the marker was lost by update_state
        return review_payload(state.values)  # rebuild what the underwriter needs to see
    return None


def submit_decision(graph, thread_id: str, decision: dict) -> dict:
    """Resume the paused case with the underwriter's decision."""
    return graph.invoke(Command(resume=decision), thread_config(thread_id))


def hold_for_underwriter(graph, thread_id: str, text: str) -> None:
    """Attach a (redacted) applicant message to a case under review, without running the graph."""
    # avoid circular import
    msg = redact_at_boundary(text)
    graph.update_state(
        thread_config(thread_id), {"messages": [msg], "held_messages": [msg.content]}
    )
