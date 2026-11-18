"""Rules for an underwriter's decision on a referred case."""

from typing import Any

ALLOWED_DECISIONS = ("approve", "decline")


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
