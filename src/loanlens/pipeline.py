"""Lesson 4: the first end-to-end LoanLens graph. Conditional edges do the routing.

  START -> extract_details -> validate_application --(route_after_validation)--+
                                                                                |
        incomplete:  request_missing_info -> END                                |
        complete:    check_affordability ┐                                      |
                     check_age           ├─> combine --(route_after_screening)--+
                     check_employment    ┘
                                                    under_18 flag -> decline              -> END
                                                    other flags   -> refer_to_underwriter -> END
                                                    no flags      -> approve_in_principle -> END

Everything except the routing comes from YOUR earlier lessons:
  extract_details (L3) · validate_application (L1) · the checks + combine (L2)

YOUR JOB:  uv run pytest tests/test_lesson04_pipeline.py -v  -> green
  1. route_after_validation   2. route_after_screening
  3. request_missing_info     4. build_graph
"""

from operator import add
from typing import Annotated, Literal  # noqa: F401

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401

from loanlens.extraction import make_extract_node  # noqa: F401
from loanlens.intake import validate_application  # noqa: F401
from loanlens.screening import check_affordability, check_age, check_employment, combine  # noqa: F401
from loanlens.pii import redact_pii

SCREENING_CHECKS = ["check_affordability", "check_age", "check_employment"]


class PipelineState(MessagesState, total=False):
    application: dict
    missing_fields: list[str]
    is_complete: bool
    risk_flags: Annotated[list[str], add]
    decision_hint: str
    outcome: str  # "awaiting_information" | "declined" | "referred" | "approved_in_principle"


# ── Routers: read state, return WHERE to go next. They never update state. ──


def route_after_validation(state: PipelineState) -> str | list[str]:
    """Incomplete application -> "request_missing_info".
    Complete application      -> SCREENING_CHECKS (a LIST of node names: all of
                                 them run next, in parallel, the same fan-out as Lesson 2)."""
    is_complete = state.get("is_complete")
    return "request_missing_info" if not is_complete else SCREENING_CHECKS


def route_after_screening(
    state: PipelineState,
) -> Literal["decline", "refer_to_underwriter", "approve_in_principle"]:
    """Policy v0:
    "under_18" in risk_flags -> "decline"   (hard rule: we never lend to minors)
    any other flag           -> "refer_to_underwriter"
    no flags                 -> "approve_in_principle\""""
    risk_flags = state.get("risk_flags")
    if "under_18" in risk_flags:
        return "decline"
    if risk_flags:
        return "refer_to_underwriter"
    return "approve_in_principle"


# ── Outcome nodes ──────────────────────────────────────────────────────────


def request_missing_info(state: PipelineState) -> dict:
    """Return:
    "outcome":  "awaiting_information"
    "messages": [AIMessage("Thanks. To continue, we need: age, annual_income.")]
                (list the missing_fields in order, comma-separated, exactly like that)"""
    missing_fields = state.get("missing_fields")
    return {
        "outcome": "awaiting_information",
        "messages": [
            AIMessage(f"Thanks. To continue, we need: {', '.join(missing_fields)}.")
        ],
    }


def decline(state: PipelineState) -> dict:
    return {
        "outcome": "declined",
        "messages": [
            AIMessage("We're unable to offer a loan: applicants must be 18 or over.")
        ],
    }


def refer_to_underwriter(state: PipelineState) -> dict:
    return {
        "outcome": "referred",
        "messages": [
            AIMessage(
                "Thanks. Your application has been passed to an underwriter for review."
            )
        ],
    }


def approve_in_principle(state: PipelineState) -> dict:
    return {
        "outcome": "approved_in_principle",
        "messages": [
            AIMessage(
                "Good news: your loan is approved in principle, subject to final checks."
            )
        ],
    }


def build_graph(model=None):
    """Nodes: extract_details (make_extract_node(model)), validate_application,
    the 3 checks, combine, request_missing_info, decline, refer_to_underwriter,
    approve_in_principle.

    Fixed edges:   START -> extract_details -> validate_application
                   each check -> combine;   each outcome node -> END
    Conditional:   builder.add_conditional_edges("validate_application",
                       route_after_validation, ["request_missing_info", *SCREENING_CHECKS])
                   (that 3rd argument lists every possible destination; LangGraph
                    needs it to draw the graph when the router can return a list)
                   ...and one more add_conditional_edges after "combine".

    If model is None, use get_model() (import inside the function, like Lesson 3)."""
    if not model:
        from loanlens.llm import get_model
        model = get_model()
    builder = StateGraph(PipelineState)
    builder.add_node(make_extract_node(model))
    builder.add_node(redact_pii)
    builder.add_node(validate_application)
    builder.add_node(check_affordability)
    builder.add_node(check_employment)
    builder.add_node(check_age)
    builder.add_node(combine)
    builder.add_node(request_missing_info)
    builder.add_node(decline)
    builder.add_node(refer_to_underwriter)
    builder.add_node(approve_in_principle)

    builder.add_edge(START,"redact_pii")
    builder.add_edge("redact_pii","extract_details")
    builder.add_edge("extract_details","validate_application")
    #builder.add_edge("validate_application","check_affordability")
    #builder.add_edge("validate_application","check_employment")
    #builder.add_edge("validate_application","check_age")
    builder.add_conditional_edges("validate_application",route_after_validation,["request_missing_info", *SCREENING_CHECKS])
    builder.add_edge("check_affordability","combine")
    builder.add_edge("check_employment","combine")
    builder.add_edge("check_age","combine")
    builder.add_conditional_edges("combine",route_after_screening)
    builder.add_edge("request_missing_info",END)
    builder.add_edge("decline",END)
    builder.add_edge("refer_to_underwriter",END)
    builder.add_edge("approve_in_principle",END)
    
    

    graph = builder.compile()
    
    return graph


