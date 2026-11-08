"""The state of one loan case, shared by every node in the pipeline.

LangGraph merges each node's return value into this state. Keys annotated with a reducer
(`Annotated[type, reducer]`) are combined with that function instead of being replaced, which is
what lets parallel nodes write to the same key in one step.
"""

from operator import add
from typing import Annotated, Any

from langgraph.graph import MessagesState


def merge_reports(left: dict | None, right: dict | None) -> dict:
    """Reducer for specialist_reports: combine reports written by parallel specialists."""
    return {**(left or {}), **(right or {})}


class PipelineState(MessagesState, total=False):
    """`messages` (the conversation) comes from MessagesState; total=False makes every key optional."""

    application: dict[str, Any]  # the redacted, extracted ApplicationDetails as a plain dict
    missing_fields: list[str]
    is_complete: bool
    risk_flags: Annotated[list[str], add]  # the parallel screening checks append here
    # "in_progress" | "awaiting_information" | "declined" | "referred"
    # | "approved_in_principle" | "declined_by_underwriter"
    outcome: str
    underwriter_decision: dict[str, Any]
    review_error: str | None
    held_messages: Annotated[list[str], add]  # applicant messages received during review
    specialist_reports: Annotated[dict[str, Any], merge_reports]
