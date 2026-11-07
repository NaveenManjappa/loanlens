"""Lesson 8: LoanLens cases that survive between messages (and restarts).

A CASE = a thread. Every message for the same applicant goes to the same thread_id,
the checkpointer saves the state after every step, and the next invoke on that thread
starts from the saved state: same messages, same application...

...and the same leftovers. Per-run scratch keys (risk_flags, outcome, decision_hint)
must be reset at the start of each run, or yesterday's flags leak into today's decision.

YOUR JOB:  uv run pytest tests/test_lesson08_persistence.py -v  -> green
  1. start_run           (a node: add it to pipeline.py as the FIRST node)
  2. continue_application
  3. current_outcome
  4. decisions_made
  + pipeline.build_graph(model=None, checkpointer=None) -> compile(checkpointer=checkpointer)
"""

from textwrap import wrap

from langgraph.types import Overwrite
import openai  # noqa: F401

from loanlens.pii import redact_at_boundary# noqa: F401
from langchain_core.messages import AIMessage, HumanMessage  # noqa: F401


def start_run(state) -> dict:
    """First node of every run. Reset the per-run scratch keys, KEEP the conversation:
        "risk_flags":    Overwrite([])   (it has an append reducer: [] alone would add nothing! Lesson 2)
        "decision_hint": None
        "outcome":       "in_progress"
    Never touch "messages": that's the case's memory."""
    return {
        "risk_flags":Overwrite([]),
        "outcome":"in_progress",
        "decision_hint":None,
        "held_messages":Overwrite([]),
        "specialist_reports":Overwrite({})
    }


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# ── Lesson 12: a message blocked by the content filter (ADR 0002) ──────────

BLOCKED_REPLY = (
    "Sorry, we couldn't process that message. "
    "Please tell us about your loan application and we'll carry on from there."
)
WITHHELD = "[message withheld]"


def is_content_filter_error(exc: BaseException) -> bool:
    """True if exc means Azure's content filter refused the request:
      - a blocked PROMPT:     an error whose .code == "content_filter" (Azure's HTTP 400)
                              (use getattr(exc, "code", None): most exceptions have no .code)
      - a blocked COMPLETION: an openai.ContentFilterFinishReasonError
    Anything else -> False. Never swallow an error you don't understand."""
    return (
        getattr(exc, "code", None) == "content_filter"
        or isinstance(exc, openai.ContentFilterFinishReasonError)
    )


def withhold_blocked_message(graph, thread_id: str) -> None:
    """The run crashed, but the blocked message is ALREADY in the checkpoint. Repair the case:
      1. find the LAST HumanMessage in graph.get_state(config).values["messages"]
      2. graph.update_state(config,
             {"messages": [HumanMessage(WITHHELD, id=<its id>), AIMessage(BLOCKED_REPLY)],
              "outcome": "awaiting_information"},
             as_node="request_missing_info")
    Same id -> add_messages REPLACES the blocked message (the Lesson 7 trick).
    as_node="request_missing_info" records the update as if LoanLens had asked for information,
    so the case ends cleanly (.next == ()) and the next message starts a normal run."""
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
    """Send one applicant message into their case and return the final state.
    Redact at the BOUNDARY (Lesson 7) so raw PII never reaches a checkpoint:
        graph.invoke({"messages": [redact_at_boundary(text)]}, thread_config(thread_id))"""
    # wrap the graph.invoke(...) in try/except Exception as exc:
    #    content-filter error -> withhold_blocked_message(graph, thread_id), then return
    #                            graph.get_state(thread_config(thread_id)).values
    #    anything else        -> raise   (re-raise it unchanged)
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
    """Read the case WITHOUT running anything: graph.get_state(config).values.get("outcome")."""
    return graph.get_state(thread_config(thread_id)).values.get("outcome")


def decisions_made(graph, thread_id: str) -> list[str]:
    """The case's audit trail: the outcome of every COMPLETED run, oldest first.
    graph.get_state_history(config) yields snapshots NEWEST first. A snapshot whose
    .next is empty () is the end of a run. Collect .values["outcome"] from those, then reverse."""
    outcomes = []
    for snapshot in graph.get_state_history(thread_config(thread_id)):
        if not snapshot.next:
            outcomes.append(snapshot.values["outcome"])
    return list(reversed(outcomes))
    
