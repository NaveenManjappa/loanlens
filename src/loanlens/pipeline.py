"""The LoanLens graph.

    start_run -> redact_pii -> extract_details -> ground_age -> validate_application
        incomplete -> request_missing_info
        complete   -> check_affordability | check_age | check_employment | check_sanctions -> combine
            under_18 flag -> decline
            other flags   -> refer_to_underwriter -> specialists (if any) -> underwriter_review
            no flags      -> approve_in_principle

build_graph() compiles it. Its dependencies are passed in: the app passes the real model, the real
specialists and a durable checkpointer; tests pass fake models and usually no specialists.
"""

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Overwrite

from loanlens.extraction import make_extract_node
from loanlens.grounding import ground_age
from loanlens.pii import redact_pii
from loanlens.screening import (
    check_affordability,
    check_age,
    check_employment,
    check_sanctions,
    combine,
    validate_application,
)
from loanlens.specialists import Specialists, make_specialist_node
from loanlens.state import PipelineState
from loanlens.underwriting import route_after_review, underwriter_decline, underwriter_review

SCREENING_CHECKS = ["check_affordability", "check_age", "check_employment", "check_sanctions"]


def start_run(state: PipelineState) -> dict:
    """First node of every run: reset the per-run keys and keep the conversation.

    Keys with a reducer are reset with Overwrite: a plain empty value would be merged into the old
    one by the reducer instead of replacing it.
    """
    return {
        "risk_flags": Overwrite([]),
        "outcome": "in_progress",
        "held_messages": Overwrite([]),
        "specialist_reports": Overwrite({}),
    }


# ── Routers: read the state and return where to go next. They never change state. ──


def route_after_validation(state: PipelineState) -> str | list[str]:
    """Incomplete -> request_missing_info; complete -> every screening check, in parallel."""
    if not state.get("is_complete"):
        return "request_missing_info"
    return SCREENING_CHECKS


def route_after_screening(
    state: PipelineState,
) -> Literal["decline", "refer_to_underwriter", "approve_in_principle"]:
    """under_18 -> decline (never lend to minors); any other flag -> a human; no flags -> approve."""
    risk_flags = state["risk_flags"]
    if "under_18" in risk_flags:
        return "decline"
    if risk_flags:
        return "refer_to_underwriter"
    return "approve_in_principle"


# ── Outcome nodes ──────────────────────────────────────────────────────────


def request_missing_info(state: PipelineState) -> dict:
    """Ask the applicant for the missing fields, in order."""
    missing = ", ".join(state["missing_fields"])
    return {
        "outcome": "awaiting_information",
        "messages": [AIMessage(f"Thanks. To continue, we need: {missing}.")],
    }


def decline(state: PipelineState) -> dict:
    return {
        "outcome": "declined",
        "messages": [AIMessage("We're unable to offer a loan: applicants must be 18 or over.")],
    }


def refer_to_underwriter(state: PipelineState) -> dict:
    return {
        "outcome": "referred",
        "messages": [AIMessage("Thanks. Your application has been passed to an underwriter for review.")],
    }


def approve_in_principle(state: PipelineState) -> dict:
    return {
        "outcome": "approved_in_principle",
        "messages": [AIMessage("Good news: your loan is approved in principle, subject to final checks.")],
    }


# ── The graph ──────────────────────────────────────────────────────────────


def build_graph(
    model: BaseChatModel,
    checkpointer: BaseCheckpointSaver | None = None,
    specialists: Specialists | None = None,
) -> CompiledStateGraph:
    """Compile the pipeline.

    model:        the chat model used for extraction
    checkpointer: where cases are saved; without one, nothing is remembered between runs
    specialists:  agents that report to the underwriter on referred cases (see specialists.py)
    """
    builder = StateGraph(PipelineState)

    builder.add_node(start_run)
    builder.add_node(redact_pii)
    builder.add_node(make_extract_node(model))
    builder.add_node(ground_age)
    builder.add_node(validate_application)
    builder.add_node(request_missing_info)
    for check in (check_affordability, check_age, check_employment, check_sanctions):
        builder.add_node(check)
    builder.add_node(combine)
    builder.add_node(decline)
    builder.add_node(approve_in_principle)
    builder.add_node(refer_to_underwriter)
    builder.add_node(underwriter_review)
    builder.add_node(underwriter_decline)

    # Intake: always the same steps, in order.
    builder.add_edge(START, "start_run")
    builder.add_edge("start_run", "redact_pii")
    builder.add_edge("redact_pii", "extract_details")
    builder.add_edge("extract_details", "ground_age")
    builder.add_edge("ground_age", "validate_application")

    # Screening: fan out to every check, then join at combine.
    builder.add_conditional_edges(
        "validate_application", route_after_validation, ["request_missing_info", *SCREENING_CHECKS]
    )
    for check_name in SCREENING_CHECKS:
        builder.add_edge(check_name, "combine")
    builder.add_conditional_edges("combine", route_after_screening)

    # Referral: specialists run in parallel, then the underwriter reviews.
    if specialists:
        for name, (agent, verdict) in specialists.items():
            node_name = f"{name}_specialist"
            builder.add_node(node_name, make_specialist_node(name, agent, verdict))
            builder.add_edge("refer_to_underwriter", node_name)
            builder.add_edge(node_name, "underwriter_review")
    else:
        builder.add_edge("refer_to_underwriter", "underwriter_review")
    builder.add_conditional_edges("underwriter_review", route_after_review)

    # Every outcome ends the run.
    for outcome_node in ("request_missing_info", "decline", "approve_in_principle", "underwriter_decline"):
        builder.add_edge(outcome_node, END)

    return builder.compile(checkpointer=checkpointer)
