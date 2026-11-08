"""Messages that arrive while a review is pending are held for the underwriter, not run (ADR 0001)."""

from langgraph.checkpoint.memory import InMemorySaver

from loanlens.cases import continue_application, current_outcome, decisions_made
from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import build_graph
from loanlens.underwriting import awaiting_underwriter, submit_decision

SAM = ApplicationDetails(
    applicant_name="Sam Jones", age=29, annual_income=40_000, loan_amount=5_000, employment_status="self_employed"
)
SAM_EMPLOYED = SAM.model_copy(update={"employment_status": "employed"})
APPROVE = {"decision": "approve", "reason": "Job offer letter verified", "underwriter_id": "uw-017"}
UPDATE = "I've just got a salaried job!"


def paused_case(make_fake, answers):
    fake = make_fake(answers)
    graph = build_graph(fake, checkpointer=InMemorySaver())
    continue_application(graph, "case-sam", "Sam, 29, self-employed, £40k, want £5,000")
    assert awaiting_underwriter(graph, "case-sam") is not None
    return graph, fake


def test_message_during_review_is_held_not_run(scripted_structured_model):
    graph, fake = paused_case(scripted_structured_model, [SAM, SAM_EMPLOYED])
    result = continue_application(graph, "case-sam", UPDATE)
    assert len(fake.all_sent) == 1, "no new run: the extractor must not be called again"
    assert result["outcome"] == "referred"
    assert current_outcome(graph, "case-sam") == "referred"
    payload = awaiting_underwriter(graph, "case-sam")
    assert payload is not None, "the case must stay in the underwriter's queue"
    assert payload["applicant_updates"] == [UPDATE]


def test_underwriter_decides_with_the_update_in_view(scripted_structured_model):
    graph, _ = paused_case(scripted_structured_model, [SAM, SAM_EMPLOYED])
    continue_application(graph, "case-sam", UPDATE)
    final = submit_decision(graph, "case-sam", APPROVE)
    assert final["outcome"] == "approved_in_principle"
    assert decisions_made(graph, "case-sam") == ["approved_in_principle"], "one run, nothing abandoned"


def test_held_messages_are_redacted(scripted_structured_model):
    graph, _ = paused_case(scripted_structured_model, [SAM])
    continue_application(graph, "case-sam", "New job! My NI is AB 12 34 56 C")
    assert "AB 12 34 56 C" not in str(awaiting_underwriter(graph, "case-sam"))


def test_several_updates_are_all_shown(scripted_structured_model):
    graph, _ = paused_case(scripted_structured_model, [SAM])
    continue_application(graph, "case-sam", "update one")
    continue_application(graph, "case-sam", "update two")
    assert awaiting_underwriter(graph, "case-sam")["applicant_updates"] == ["update one", "update two"]


def test_after_the_decision_new_messages_start_a_fresh_run(scripted_structured_model):
    graph, fake = paused_case(scripted_structured_model, [SAM, SAM_EMPLOYED])
    continue_application(graph, "case-sam", UPDATE)
    submit_decision(graph, "case-sam", APPROVE)
    result = continue_application(graph, "case-sam", "Can I borrow a bit more?")
    assert len(fake.all_sent) == 2, "not paused any more -> a normal run"
    assert result["held_messages"] == [], "start_run clears the held updates"


def test_cases_not_under_review_behave_as_before(scripted_structured_model):
    fake = scripted_structured_model([SAM_EMPLOYED])
    graph = build_graph(fake, checkpointer=InMemorySaver())
    result = continue_application(graph, "case-new", "Sam, 29, employed, £40k, want £5,000")
    assert result["outcome"] == "approved_in_principle" and len(fake.all_sent) == 1
