"""The evaluators, and sanctions screening for every applicant."""

from langgraph.checkpoint.memory import InMemorySaver

from loanlens.cases import continue_application
from loanlens.evals import (
    age_not_invented,
    extraction_correct,
    load_dataset,
    no_raw_pii_sent_to_models,
    outcome_matches,
    summarise,
)
from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import build_graph


def run(outcome=None, application=None, prompts=()):
    return {"outcome": outcome, "application": application or {}, "prompts": list(prompts)}


# ── Part A: evaluators ─────────────────────────────────────────────────────


def test_dataset_loads():
    ids = [c["id"] for c in load_dataset()]
    assert "sanctioned-person" in ids and len(ids) == len(set(ids))


def test_dataset_has_a_job_title_only_case():
    case = next(c for c in load_dataset() if c["id"] == "job-title-only")
    assert case["expect_outcome"] == "awaiting_information"
    assert case["expect_employment_status"] is None


def test_other_cases_state_employment():
    """One case, one thing tested: only job-title-only leaves the employment status unstated."""
    words = ("employed", "self-employed", "freelance", "retired", "unemployed", "student")
    for case in load_dataset():
        if case["id"] != "job-title-only":
            assert any(w in case["text"].lower() for w in words), f"{case['id']} doesn't state employment"


def test_outcome_matches():
    assert outcome_matches({"expect_outcome": "declined"}, run("declined")) is True
    assert outcome_matches({"expect_outcome": "declined"}, run("referred")) is False
    assert outcome_matches({"expect_not_outcome": "approved_in_principle"}, run("referred")) is True
    assert outcome_matches({"expect_not_outcome": "approved_in_principle"}, run("approved_in_principle")) is False
    assert outcome_matches({}, run("declined")) is None


def test_no_raw_pii_sent_to_models():
    case = {"pii_strings": ["AB 12 34 56 C", "41234567"]}
    assert no_raw_pii_sent_to_models(case, run(prompts=["NI [REDACTED_UK_NINO]"])) is True
    assert no_raw_pii_sent_to_models(case, run(prompts=["fine", "account 41234567"])) is False
    assert no_raw_pii_sent_to_models({}, run(prompts=["anything"])) is None


def test_age_not_invented():
    case = {"text": "I'm Priya, 34, nurse"}
    assert age_not_invented(case, run(application={"age": 34})) is True
    assert age_not_invented(case, run(application={"age": None})) is True
    assert age_not_invented(case, run(application={"age": 30})) is False, "30 appears nowhere: invented"
    assert age_not_invented({"text": "aged 134 years"}, run(application={"age": 34})) is False, "whole number only"


def test_extraction_correct():
    case = {"expect_outcome": "x", "expect_age": 34, "expect_annual_income": 42000}
    assert extraction_correct(case, run(application={"age": 34, "annual_income": 42000.0})) is True
    assert extraction_correct(case, run(application={"age": 34, "annual_income": 3500.0})) is False
    assert extraction_correct({"expect_age": None}, run(application={"age": None})) is True
    assert extraction_correct({"expect_outcome": "x"}, run(application={"age": 1})) is None


def test_summarise_pass_rates_ignore_not_applicable():
    results = [
        {"case_id": "a", "scores": {"outcome_matches": True, "no_raw_pii_sent_to_models": None}},
        {"case_id": "a", "scores": {"outcome_matches": False, "no_raw_pii_sent_to_models": None}},
        {"case_id": "b", "scores": {"outcome_matches": True, "no_raw_pii_sent_to_models": True}},
    ]
    assert summarise(results) == {"outcome_matches": 0.67, "no_raw_pii_sent_to_models": 1.0}


# ── Part B: the sanctions fix ──────────────────────────────────────────────


def test_sanctioned_applicant_is_never_auto_approved(scripted_structured_model):
    ivan = ApplicationDetails(
        applicant_name="Ivan Petrov", age=45, annual_income=70_000, loan_amount=10_000, employment_status="employed"
    )
    graph = build_graph(scripted_structured_model([ivan]), checkpointer=InMemorySaver())
    final = continue_application(graph, "case-ivan", "Ivan Petrov, 45, employed, £70k, want £10k")
    assert "sanctions_match" in final["risk_flags"]
    assert final["outcome"] == "referred", "a human (compliance) must confirm a name match"


def test_everyone_is_screened(scripted_structured_model):
    priya = ApplicationDetails(
        applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
    )
    graph = build_graph(scripted_structured_model([priya]), checkpointer=InMemorySaver())
    visited = []
    for step in graph.stream(
        {"messages": [("user", "Priya Shah, 34, employed nurse, £42,000 a year, want £8,000")]}, {"configurable": {"thread_id": "p"}}, stream_mode="updates"
    ):
        visited.extend(step)
    assert "check_sanctions" in visited, "sanctions screening must run for clean applicants too"
