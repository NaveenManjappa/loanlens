"""The LoanLens graph.

    start_run -> redact_pii -> extract_details -> ground_age -> validate_application
        incomplete -> request_missing_info
        complete   -> check_affordability | check_age | check_employment | check_sanctions -> combine
            under_18 flag -> decline
            other flags   -> refer_to_underwriter -> risk + compliance specialists -> underwriter_review
            no flags      -> approve_in_principle

build_graph(model=None, checkpointer=None, specialists=None) compiles it: tests inject fake models,
the app uses the real model, the default specialists and a durable checkpointer.
"""

from operator import add
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from loanlens.extraction import make_extract_node
from loanlens.grounding import ground_age
from loanlens.intake import validate_application
from loanlens.screening import check_affordability, check_age, check_employment, check_sanctions, combine
from loanlens.pii import redact_pii
from loanlens.cases import start_run
from loanlens.specialists import default_specialists, make_specialist_node, merge_reports
from loanlens.underwriting import route_after_review, underwriter_decline, underwriter_review
SCREENING_CHECKS = ["check_affordability", "check_age", "check_employment","check_sanctions"]


class PipelineState(MessagesState, total=False):
    application: dict
    missing_fields: list[str]
    is_complete: bool
    risk_flags: Annotated[list[str], add]
    decision_hint: str
    outcome: str  # "awaiting_information" | "declined" | "referred" | "approved_in_principle"
    underwriter_decision:dict
    review_error:str | None
    held_messages: Annotated[list[str], add]
    specialist_reports: Annotated[dict, merge_reports]

# ── Routers: read state, return WHERE to go next. They never update state. ──


def route_after_validation(state: PipelineState) -> str | list[str]:
    """Incomplete -> request_missing_info; complete -> every screening check, in parallel."""
    is_complete = state.get("is_complete")
    return "request_missing_info" if not is_complete else SCREENING_CHECKS


def route_after_screening(
    state: PipelineState,
) -> Literal["decline", "refer_to_underwriter", "approve_in_principle"]:
    """Policy: under_18 -> decline (never lend to minors); any other flag -> refer_to_underwriter;
    no flags -> approve_in_principle.
    """
    risk_flags = state.get("risk_flags")
    if "under_18" in risk_flags:
        return "decline"
    if risk_flags:
        return "refer_to_underwriter"
    return "approve_in_principle"


# ── Outcome nodes ──────────────────────────────────────────────────────────


def request_missing_info(state: PipelineState) -> dict:
    """Ask the applicant for the missing fields, in order."""
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


def build_graph(model=None,checkpointer=None,specialists=None):
    """Compile the pipeline.

    Without a model, the real one is used. Specialists default to the real agents only when the model
    is real; a test that passes a fake model gets none.
    """
    use_default_specialists = model is None
    if model is None:
        from loanlens.llm import get_model
        model = get_model()
    if specialists is None:
        specialists = default_specialists(model) if use_default_specialists else {}


    builder = StateGraph(PipelineState)
    builder.add_node(make_extract_node(model))
    builder.add_node(start_run)
    builder.add_node(redact_pii)
    builder.add_node(validate_application)
    builder.add_node(check_affordability)
    builder.add_node(ground_age)
    builder.add_node(check_employment)
    builder.add_node(check_sanctions)
    builder.add_node(check_age)
    builder.add_node(combine)
    builder.add_node(request_missing_info)
    builder.add_node(decline)
    builder.add_node(refer_to_underwriter)
    builder.add_node(approve_in_principle)
    builder.add_node(underwriter_review)
    builder.add_node(underwriter_decline)

    

    builder.add_edge(START,"start_run")
    builder.add_edge("start_run","redact_pii")
    builder.add_edge("redact_pii","extract_details")
    builder.add_edge("extract_details","ground_age")
    builder.add_edge("ground_age","validate_application")
    builder.add_conditional_edges("validate_application",route_after_validation,["request_missing_info", *SCREENING_CHECKS])
    builder.add_edge("check_affordability","combine")
    builder.add_edge("check_employment","combine")
    builder.add_edge("check_age","combine")
    builder.add_edge("check_sanctions","combine")
    builder.add_conditional_edges("combine",route_after_screening)
    builder.add_edge("request_missing_info",END)
    builder.add_edge("decline",END)
    if specialists:
        for name, (agent, verdict) in specialists.items():
            builder.add_node(f"{name}_specialist", make_specialist_node(name, agent, verdict))
            builder.add_edge("refer_to_underwriter",f"{name}_specialist")
            builder.add_edge(f"{name}_specialist","underwriter_review")
    else:
        builder.add_edge("refer_to_underwriter","underwriter_review")
    builder.add_conditional_edges("underwriter_review",route_after_review)
    builder.add_edge("underwriter_decline",END)
    builder.add_edge("approve_in_principle",END)
    
    

    graph = builder.compile(checkpointer=checkpointer)
    
    return graph

