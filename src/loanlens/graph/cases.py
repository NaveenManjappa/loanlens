"""Operations on a case. A case is a LangGraph thread: every message for the same applicant goes to
the same thread_id, and the checkpointer saves the state after every step, so the next message
continues from the saved case.

    continue_application   an applicant message: run the pipeline, or hold it during a review
    awaiting_underwriter   the pending review, if the case is paused
    submit_decision        resume a paused case with the underwriter's decision

This module also recovers a case after the provider's content filter blocks a message (ADR 0002).
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from loanlens.graph.intake import redact_at_boundary
from loanlens.graph.underwriting import review_payload
from loanlens.integrations.llm import is_content_filter_error

logger = logging.getLogger(__name__)

BLOCKED_REPLY = (
    "Sorry, we couldn't process that message. "
    "Please tell us about your loan application and we'll carry on from there."
)
WITHHELD = "[message withheld]"


def thread_config(case_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": case_id}}


# ── Reading a case ─────────────────────────────────────────────────────────


def awaiting_underwriter(graph: CompiledStateGraph, case_id: str) -> dict[str, Any] | None:
    """The review payload if the case is paused for an underwriter, otherwise None.

    A paused case whose interrupt marker was cleared by update_state (hold_for_underwriter does
    this) still has underwriter_review as its next step, so it's rebuilt from the state.
    """
    snapshot = graph.get_state(thread_config(case_id))
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    if snapshot.next == ("underwriter_review",):
        return review_payload(snapshot.values)
    return None


# ── Changing a case ────────────────────────────────────────────────────────


def continue_application(graph: CompiledStateGraph, case_id: str, text: str) -> dict[str, Any]:
    """Send one applicant message into their case and return the case state.

    The text is redacted before it enters the graph, so raw PII never reaches a checkpoint. While a
    review is pending, the message is held for the underwriter instead of starting a new run.
    """
    if awaiting_underwriter(graph, case_id) is not None:
        hold_for_underwriter(graph, case_id, text)
        return graph.get_state(thread_config(case_id)).values

    try:
        state = graph.invoke({"messages": [redact_at_boundary(text)]}, thread_config(case_id))
    except Exception as exc:
        if not is_content_filter_error(exc):
            raise
        logger.warning("case %s: message blocked by the content filter and withheld", case_id)
        withhold_blocked_message(graph, case_id)
        return graph.get_state(thread_config(case_id)).values

    logger.info("case %s: run finished with outcome %s", case_id, state.get("outcome"))
    return state


def hold_for_underwriter(graph: CompiledStateGraph, case_id: str, text: str) -> None:
    """Attach a (redacted) applicant message to a case under review, without running the graph (ADR 0001)."""
    message = redact_at_boundary(text)
    graph.update_state(
        thread_config(case_id),
        {"messages": [message], "held_messages": [message.content]},
    )
    logger.info("case %s: applicant message held for the underwriter", case_id)


def submit_decision(graph: CompiledStateGraph, case_id: str, decision: dict[str, Any]) -> dict[str, Any]:
    """Resume the paused case with the underwriter's decision (underwriting.underwriter_review validates it)."""
    logger.info(
        "case %s: underwriter %s submitted decision %s",
        case_id,
        decision.get("underwriter_id"),
        decision.get("decision"),
    )
    return graph.invoke(Command(resume=decision), thread_config(case_id))


# ── Content-filter recovery (ADR 0002) ─────────────────────────────────────


def withhold_blocked_message(graph: CompiledStateGraph, case_id: str) -> None:
    """Repair a case after a content-filter block.

    The blocked message is already checkpointed, so every later run would resend it and be blocked
    again. It's replaced by a placeholder with the same message id (add_messages replaces a message
    with a matching id instead of appending), a fixed reply is added, and the update is recorded as
    request_missing_info, so the run ends cleanly and the next message starts a normal run.
    """
    messages = graph.get_state(thread_config(case_id)).values["messages"]
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if not human_messages:
        raise ValueError(f"case {case_id}: no applicant message to withhold")
    blocked_id = human_messages[-1].id

    graph.update_state(
        thread_config(case_id),
        {
            "messages": [HumanMessage(WITHHELD, id=blocked_id), AIMessage(BLOCKED_REPLY)],
            "outcome": "awaiting_information",
        },
        as_node="request_missing_info",
    )
