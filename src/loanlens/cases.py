"""A case is a LangGraph thread.

Every message for the same applicant goes to the same thread_id. The checkpointer saves the state
after every step, so the next message continues from the saved case. start_run resets the per-run
keys, so one run's flags never leak into the next. This module also recovers a case after the
provider's content filter blocks a message (ADR 0002).
"""


from langgraph.types import Overwrite
import openai

from loanlens.pii import redact_at_boundary
from langchain_core.messages import AIMessage, HumanMessage


def start_run(state) -> dict:
    """First node of every run: reset the per-run keys and keep the conversation.

    Keys with a reducer are reset with Overwrite: returning a plain empty value would be merged
    into the old one instead of replacing it.
    """
    return {
        "risk_flags":Overwrite([]),
        "outcome":"in_progress",
        "decision_hint":None,
        "held_messages":Overwrite([]),
        "specialist_reports":Overwrite({})
    }


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ── Content-filter recovery (ADR 0002) ─────────────────────────────────────

BLOCKED_REPLY = (
    "Sorry, we couldn't process that message. "
    "Please tell us about your loan application and we'll carry on from there."
)
WITHHELD = "[message withheld]"


def is_content_filter_error(exc: BaseException) -> bool:
    """True if exc means the provider's content filter refused the request.

    That is a blocked prompt (an error whose .code is "content_filter", Azure's HTTP 400) or a
    blocked completion (openai.ContentFilterFinishReasonError). Anything else is re-raised by the caller.
    """
    return (
        getattr(exc, "code", None) == "content_filter"
        or isinstance(exc, openai.ContentFilterFinishReasonError)
    )


def withhold_blocked_message(graph, thread_id: str) -> None:
    """Repair a case after a content-filter block.

    The blocked message is already checkpointed, and every later run would resend it and be blocked
    again. It's replaced by a placeholder with the same message id (so add_messages replaces it rather
    than appending), a fixed reply is added, and the update is recorded as request_missing_info, so the
    run ends cleanly and the next message starts a normal run.
    """
    state = graph.get_state(thread_config(thread_id)).values
    messages = state["messages"]
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            blocked_id = m.id
            break
    else:
        raise ValueError("No HumanMessage found in the case's messages")
    graph.update_state(
        thread_config(thread_id),
        {
            "messages": [
                HumanMessage(WITHHELD, id=blocked_id),
                AIMessage(BLOCKED_REPLY),
            ],
            "outcome": "awaiting_information",
        },
        as_node="request_missing_info",
    )


def continue_application(graph, thread_id: str, text: str) -> dict:
    """Send one applicant message into their case and return the case state.

    The text is redacted before it enters the graph, so raw PII never reaches a checkpoint. While a
    review is pending, the message is held for the underwriter instead of starting a new run.
    """
    from loanlens.underwriting import awaiting_underwriter  # avoid circular import
    if awaiting_underwriter(graph, thread_id):
        from loanlens.underwriting import hold_for_underwriter  # avoid circular import
        hold_for_underwriter(graph, thread_id, text)
        return graph.get_state(thread_config(thread_id)).values
    
    try:
        return graph.invoke(
            {"messages": [redact_at_boundary(text)]},
            thread_config(thread_id),
        )
    except Exception as exc:
        if not is_content_filter_error(exc):
            raise
        withhold_blocked_message(graph, thread_id)
        return graph.get_state(thread_config(thread_id)).values

def current_outcome(graph, thread_id: str) -> str | None:
    """The case's current outcome, read without running anything."""
    return graph.get_state(thread_config(thread_id)).values.get("outcome")


def decisions_made(graph, thread_id: str) -> list[str]:
    """The outcome of every completed run of the case, oldest first."""
    outcomes = []
    for snapshot in graph.get_state_history(thread_config(thread_id)):
        if not snapshot.next:
            outcomes.append(snapshot.values["outcome"])
    return list(reversed(outcomes))
    
