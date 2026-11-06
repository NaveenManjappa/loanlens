"""Lesson 4 spec. Run: uv run pytest tests/test_lesson04_pipeline.py -v

All graph tests use the fake model from conftest.py, so they're free and offline.
"""

from langchain_core.messages import HumanMessage

from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import (
    SCREENING_CHECKS,
    build_graph,
    request_missing_info,
    route_after_screening,
    route_after_validation,
)

CLEAN = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=50_000, loan_amount=10_000, employment_status="employed"
)
NO_AGE = CLEAN.model_copy(update={"age": None})
MINOR = CLEAN.model_copy(update={"age": 17})
SELF_EMPLOYED = CLEAN.model_copy(update={"employment_status": "self_employed"})
MSG = [HumanMessage("(applicant's message; the fake model ignores it)")]


def message_for(details):
    """A realistic applicant message that says what the fake extractor 'finds'.
    (Since Lesson 11's age guard, an age the applicant never typed is thrown away.)"""
    age = f", {details.age}" if details.age is not None else ""
    return [HumanMessage(f"I'm {details.applicant_name}{age}, {details.employment_status}, "
                         f"£{details.annual_income:,.0f} a year, want £{details.loan_amount:,.0f}")]


def run(fake_structured_model, details):
    """Run the pipeline and also record which nodes actually executed."""
    graph = build_graph(fake_structured_model(details))
    msg = message_for(details)
    visited = []
    for step in graph.stream({"messages": msg}, stream_mode="updates"):
        visited.extend(step.keys())
    final = graph.invoke({"messages": msg})
    return final, visited


# ── Routers are plain functions too ────────────────────────────────────────


def test_route_after_validation():
    assert route_after_validation({"is_complete": False}) == "request_missing_info"
    assert sorted(route_after_validation({"is_complete": True})) == sorted(SCREENING_CHECKS)


def test_route_after_screening():
    assert route_after_screening({"risk_flags": ["high_loan_to_income", "under_18"]}) == "decline"
    assert route_after_screening({"risk_flags": ["self_employed_needs_review"]}) == "refer_to_underwriter"
    assert route_after_screening({"risk_flags": []}) == "approve_in_principle"


def test_request_missing_info_message_is_exact():
    out = request_missing_info({"missing_fields": ["age", "annual_income"]})
    assert out["outcome"] == "awaiting_information"
    assert out["messages"][0].content == "Thanks. To continue, we need: age, annual_income."


# ── The wiring ─────────────────────────────────────────────────────────────


def test_routing_edges_are_conditional(fake_structured_model):
    edges = build_graph(fake_structured_model(CLEAN)).get_graph().edges
    conditional = {(e.source, e.target) for e in edges if e.conditional}
    assert ("validate_application", "request_missing_info") in conditional
    assert ("validate_application", "check_age") in conditional
    assert ("combine", "decline") in conditional


# ── End to end: one test per path ──────────────────────────────────────────


def test_incomplete_application_skips_screening(fake_structured_model):
    final, visited = run(fake_structured_model, NO_AGE)
    assert final["outcome"] == "awaiting_information"
    assert final["messages"][-1].content == "Thanks. To continue, we need: age."
    assert "combine" not in visited and "check_age" not in visited, "screening should never run"


def test_clean_application_is_approved_in_principle(fake_structured_model):
    final, visited = run(fake_structured_model, CLEAN)
    assert final["outcome"] == "approved_in_principle"
    assert set(SCREENING_CHECKS) <= set(visited)


def test_self_employed_is_referred(fake_structured_model):
    final, _ = run(fake_structured_model, SELF_EMPLOYED)
    assert final["outcome"] == "referred"


def test_minor_is_declined(fake_structured_model):
    final, _ = run(fake_structured_model, MINOR)
    assert final["outcome"] == "declined"
    assert final["messages"][-1].content.startswith("We're unable to offer a loan")
