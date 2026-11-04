"""Lesson 9 spec. Run: uv run pytest tests/test_lesson09_hitl.py -v

Changes to make in pipeline.py:
  1. PipelineState: add   underwriter_decision: dict   and   review_error: str | None
     (a node returning a key the state doesn't declare is SILENTLY DROPPED: Lesson 4 map!)
  2. Nodes: add underwriter_review and underwriter_decline (import from loanlens.underwriting)
  3. Edges: refer_to_underwriter -> underwriter_review   (REPLACE refer_to_underwriter -> END)
            add_conditional_edges("underwriter_review", route_after_review)
            underwriter_decline -> END
"""

from langgraph.checkpoint.memory import InMemorySaver

from loanlens.cases import continue_application, current_outcome, decisions_made
from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import build_graph
from loanlens.underwriting import (
    awaiting_underwriter,
    review_payload,
    route_after_review,
    submit_decision,
)

SAM = ApplicationDetails(
    applicant_name="Sam Jones", age=29, annual_income=40_000, loan_amount=5_000, employment_status="self_employed"
)
PRIYA = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
)
APPROVE = {"decision": "approve", "reason": "Two years of verified accounts", "underwriter_id": "uw-017"}
DECLINE = {"decision": "decline", "reason": "Income too volatile", "underwriter_id": "uw-017"}
SAM_TEXT = "Sam Jones, 29, self-employed, £40k, want £5,000. NI AB 12 34 56 C"


def referred_case(fake):
    graph = build_graph(fake, checkpointer=InMemorySaver())
    result = continue_application(graph, "case-sam", SAM_TEXT)
    return graph, result


# ── Pure functions ─────────────────────────────────────────────────────────


def test_review_payload_shows_what_the_underwriter_needs():
    p = review_payload({"application": {"age": 29}, "risk_flags": ["x"], "messages": ["..."]})
    assert p == {"application": {"age": 29}, "risk_flags": ["x"], "error": None}


def test_route_after_review():
    assert route_after_review({"review_error": "missing reason"}) == "underwriter_review"
    assert route_after_review({"review_error": None, "underwriter_decision": APPROVE}) == "approve_in_principle"
    assert route_after_review({"review_error": None, "underwriter_decision": DECLINE}) == "underwriter_decline"


# ── Pause ──────────────────────────────────────────────────────────────────


def test_referred_case_pauses_for_the_underwriter(scripted_structured_model):
    graph, result = referred_case(scripted_structured_model([SAM]))
    assert "__interrupt__" in result, "the run should stop at interrupt()"
    assert result["outcome"] == "referred"
    payload = awaiting_underwriter(graph, "case-sam")
    assert payload["risk_flags"] == ["self_employed_needs_review"]
    assert payload["application"]["applicant_name"] == "Sam Jones"
    assert "AB 12 34 56 C" not in str(payload), "the underwriter screen gets redacted data too"
    assert graph.get_state({"configurable": {"thread_id": "case-sam"}}).next == ("underwriter_review",)


def test_clean_cases_never_pause(scripted_structured_model):
    graph = build_graph(scripted_structured_model([PRIYA]), checkpointer=InMemorySaver())
    result = continue_application(graph, "case-priya", "Priya, 34, nurse, £42k, want £8k")
    assert "__interrupt__" not in result and result["outcome"] == "approved_in_principle"
    assert awaiting_underwriter(graph, "case-priya") is None


# ── Resume ─────────────────────────────────────────────────────────────────


def test_underwriter_approves(scripted_structured_model):
    fake = scripted_structured_model([SAM])
    graph, _ = referred_case(fake)
    final = submit_decision(graph, "case-sam", APPROVE)
    assert final["outcome"] == "approved_in_principle"
    assert final["underwriter_decision"] == APPROVE, "who decided and why is now in the case"
    assert len(fake.all_sent) == 1, "resuming continues the paused run: no re-extraction"
    assert awaiting_underwriter(graph, "case-sam") is None


def test_underwriter_declines(scripted_structured_model):
    graph, _ = referred_case(scripted_structured_model([SAM]))
    final = submit_decision(graph, "case-sam", DECLINE)
    assert final["outcome"] == "declined_by_underwriter"
    assert decisions_made(graph, "case-sam") == ["declined_by_underwriter"]


def test_invalid_decision_is_rejected_and_asked_again(scripted_structured_model):
    graph, _ = referred_case(scripted_structured_model([SAM]))
    again = submit_decision(graph, "case-sam", {"decision": "approve", "underwriter_id": "uw-017"})
    assert "__interrupt__" in again, "no reason given -> still paused"
    assert "reason" in awaiting_underwriter(graph, "case-sam")["error"]
    assert current_outcome(graph, "case-sam") == "referred"
    final = submit_decision(graph, "case-sam", APPROVE)
    assert final["outcome"] == "approved_in_principle"
