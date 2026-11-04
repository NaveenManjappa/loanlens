"""Lesson 9: a human underwriter signs off every referred case.

    ... combine --route_after_screening--> refer_to_underwriter -> underwriter_review
                                                                       |  interrupt(): the graph PAUSES here,
                                                                       |  saves its state, and waits (hours, days...)
                                     route_after_review <--------------+  resumed with Command(resume=decision)
                                       | invalid decision -> underwriter_review (ask again)
                                       | "approve"        -> approve_in_principle -> END
                                       | "decline"        -> underwriter_decline  -> END

Pattern from the LangGraph docs: call interrupt() EXACTLY ONCE per node run, store any
validation error in state, and loop back with a conditional edge. (Never `while True: interrupt()`.)

YOUR JOB:  uv run pytest tests/test_lesson09_hitl.py -v  -> green
  1. review_payload   2. underwriter_review   3. route_after_review
  4. awaiting_underwriter   5. submit_decision
  + pipeline.py: add 2 keys to PipelineState, 2 nodes, change refer_to_underwriter's edge (see test file top)
"""

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt  # noqa: F401

from loanlens.cases import thread_config  # noqa: F401

ALLOWED_DECISIONS = ("approve", "decline")


def review_payload(state) -> dict:
    """What the underwriter's screen shows: everything they need to decide, nothing more.
    Return {"application": ..., "risk_flags": ..., "error": state.get("review_error")}
    (the application is already PII-redacted: Lesson 7). Must be JSON-serialisable."""
    raise NotImplementedError


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
    """Node. Call interrupt(review_payload(state)) EXACTLY ONCE: the return value is
    whatever the underwriter sent via Command(resume=...).
      invalid (validate_decision gives an error) -> return {"review_error": error}
      valid                                      -> return {"underwriter_decision": decision, "review_error": None}
    Remember: on resume this node RE-RUNS FROM ITS FIRST LINE. Keep everything before
    interrupt() side-effect free (no emails, no database writes)."""
    raise NotImplementedError


def route_after_review(state) -> Literal["underwriter_review", "approve_in_principle", "underwriter_decline"]:
    """review_error set                  -> "underwriter_review"  (ask again)
    underwriter_decision["decision"] == "approve" -> "approve_in_principle"
    otherwise                            -> "underwriter_decline\""""
    raise NotImplementedError


def underwriter_decline(state) -> dict:
    return {
        "outcome": "declined_by_underwriter",
        "messages": [AIMessage("After review by our underwriting team, we're unable to offer this loan.")],
    }


# ── Helpers for the underwriter's app (no graph changes) ───────────────────


def awaiting_underwriter(graph, thread_id: str) -> dict | None:
    """If the case is paused at an interrupt, return its payload, else None.
    graph.get_state(config).tasks is a tuple of tasks; a paused task has a non-empty
    .interrupts tuple, and each Interrupt has a .value (your review_payload)."""
    raise NotImplementedError


def submit_decision(graph, thread_id: str, decision: dict) -> dict:
    """Resume the paused case with the underwriter's decision and return the result:
    graph.invoke(Command(resume=decision), thread_config(thread_id))"""
    raise NotImplementedError
