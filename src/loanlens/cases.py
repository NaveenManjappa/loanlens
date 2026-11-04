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

from langgraph.types import Overwrite  # noqa: F401

from loanlens.pii import redact_at_boundary  # noqa: F401


def start_run(state) -> dict:
    """First node of every run. Reset the per-run scratch keys, KEEP the conversation:
        "risk_flags":    Overwrite([])   (it has an append reducer: [] alone would add nothing! Lesson 2)
        "decision_hint": None
        "outcome":       "in_progress"
    Never touch "messages": that's the case's memory."""
    return {
        "risk_flags":Overwrite([]),
        "outcome":"in_progress",
        "decision_hint":None
    }


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def continue_application(graph, thread_id: str, text: str) -> dict:
    """Send one applicant message into their case and return the final state.
    Redact at the BOUNDARY (Lesson 7) so raw PII never reaches a checkpoint:
        graph.invoke({"messages": [redact_at_boundary(text)]}, thread_config(thread_id))"""
    return graph.invoke({"messages":[redact_at_boundary(text)]},thread_config(thread_id))

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
    
