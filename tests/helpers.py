"""Read-only helpers the tests use to inspect a case."""

from loanlens.cases import thread_config


def current_outcome(graph, case_id: str) -> str | None:
    """The case's current outcome, read without running anything."""
    return graph.get_state(thread_config(case_id)).values.get("outcome")


def decisions_made(graph, case_id: str) -> list[str]:
    """The outcome of every completed run of the case, oldest first."""
    outcomes = []
    for snapshot in graph.get_state_history(thread_config(case_id)):
        if not snapshot.next:  # a finished run has no next step
            outcomes.append(snapshot.values["outcome"])
    return list(reversed(outcomes))
