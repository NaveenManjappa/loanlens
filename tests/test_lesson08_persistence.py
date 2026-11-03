"""Lesson 8 spec. Run: uv run pytest tests/test_lesson08_persistence.py -v"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Overwrite

from loanlens.cases import continue_application, current_outcome, decisions_made, start_run
from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import build_graph

TOM_NO_AGE = ApplicationDetails(
    applicant_name="Tom Reid", age=None, annual_income=60_000, loan_amount=8_000, employment_status="employed"
)
TOM_COMPLETE = TOM_NO_AGE.model_copy(update={"age": 41})
SAM_SELF_EMPLOYED = ApplicationDetails(
    applicant_name="Sam Jones", age=29, annual_income=40_000, loan_amount=5_000, employment_status="self_employed"
)
SAM_NOW_EMPLOYED = SAM_SELF_EMPLOYED.model_copy(update={"employment_status": "employed"})


def pipeline(fake):
    return build_graph(fake, checkpointer=InMemorySaver())


# ── start_run ──────────────────────────────────────────────────────────────


def test_start_run_resets_scratch_but_not_messages():
    out = start_run({"risk_flags": ["under_18"], "outcome": "declined", "messages": ["keep me"]})
    assert isinstance(out["risk_flags"], Overwrite), "an append reducer needs Overwrite to really clear"
    assert out["outcome"] == "in_progress" and out["decision_hint"] is None
    assert "messages" not in out


# ── The case remembers: Tom replies with his age ───────────────────────────


def test_tom_replies_and_the_case_continues(scripted_structured_model):
    fake = scripted_structured_model([TOM_NO_AGE, TOM_COMPLETE])
    graph = pipeline(fake)

    first = continue_application(graph, "case-tom", "Tom Reid, plumber, £60k a year, want £8,000.")
    assert first["outcome"] == "awaiting_information"

    second = continue_application(graph, "case-tom", "Sorry! I'm 41.")
    assert second["outcome"] == "approved_in_principle"
    human = [m.content for m in second["messages"] if m.type == "human"]
    assert len(human) == 2, "both of Tom's messages live in the case"
    sent_second_time = " ".join(str(m.content) for m in fake.all_sent[1])
    assert "£60k" in sent_second_time and "I'm 41" in sent_second_time, "extraction re-reads the whole case"


def test_threads_are_separate_cases(scripted_structured_model):
    graph = pipeline(scripted_structured_model([TOM_NO_AGE, TOM_COMPLETE]))
    continue_application(graph, "case-tom", "Tom here, no age given")
    other = continue_application(graph, "case-someone-else", "Hello")
    assert len([m for m in other["messages"] if m.type == "human"]) == 1


def test_without_a_checkpointer_nothing_is_remembered(scripted_structured_model):
    graph = build_graph(scripted_structured_model([TOM_NO_AGE, TOM_COMPLETE]))
    graph.invoke({"messages": [("user", "first")]})
    second = graph.invoke({"messages": [("user", "second")]})
    assert [m.content for m in second["messages"] if m.type == "human"] == ["second"], "the first run is forgotten"


# ── The leftover trap ──────────────────────────────────────────────────────


def test_yesterdays_flags_do_not_leak_into_todays_decision(scripted_structured_model):
    graph = pipeline(scripted_structured_model([SAM_SELF_EMPLOYED, SAM_NOW_EMPLOYED]))
    first = continue_application(graph, "case-sam", "Sam, 29, self-employed, £40k, want £5k")
    assert first["outcome"] == "referred"
    second = continue_application(graph, "case-sam", "Update: I've taken a salaried job.")
    assert second["risk_flags"] == [], "the self-employed flag belongs to the OLD run"
    assert second["outcome"] == "approved_in_principle"


# ── Reading a case without running it ──────────────────────────────────────


def test_current_outcome_and_audit_trail(scripted_structured_model):
    graph = pipeline(scripted_structured_model([TOM_NO_AGE, TOM_COMPLETE]))
    assert current_outcome(graph, "case-tom") is None, "no case yet"
    continue_application(graph, "case-tom", "Tom, no age")
    continue_application(graph, "case-tom", "I'm 41")
    assert current_outcome(graph, "case-tom") == "approved_in_principle"
    assert decisions_made(graph, "case-tom") == ["awaiting_information", "approved_in_principle"]


def test_no_raw_pii_in_any_checkpoint(scripted_structured_model):
    graph = pipeline(scripted_structured_model([TOM_COMPLETE]))
    continue_application(graph, "case-tom", "Tom, 41, NI AB 12 34 56 C, £60k, want £8,000")
    for snapshot in graph.get_state_history({"configurable": {"thread_id": "case-tom"}}):
        assert "AB 12 34 56 C" not in str(snapshot.values), f"raw NI number at step {snapshot.metadata['step']}"
