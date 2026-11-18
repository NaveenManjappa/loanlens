"""Validation, the screening checks and the risk_flags reducer."""

import copy
from typing import get_type_hints

from langchain_core.messages import HumanMessage

from loanlens.domain.application import ApplicationDetails
from loanlens.graph.pipeline import build_graph
from loanlens.graph.screening import (
    check_affordability,
    check_age,
    check_employment,
    check_sanctions,
    combine,
    validate_application,
)
from loanlens.graph.state import PipelineState
from loanlens.integrations.compliance_registry import is_sanctioned
from loanlens.tools.compliance_tools import check_sanctions_list

CLEAN = {
    "applicant_name": "Priya Shah",
    "age": 34,
    "annual_income": 50_000,
    "loan_amount": 25_000,  # exactly 0.5 -> NOT over the limit
    "employment_status": "employed",
}
RISKY = dict(CLEAN, age=17, loan_amount=40_000, employment_status="self_employed")


# ── validate_application ───────────────────────────────────────────────────


def test_validate_complete_application():
    assert validate_application({"application": CLEAN}) == {"missing_fields": [], "is_complete": True}


def test_validate_lists_absent_and_none_fields_in_order():
    application = {k: v for k, v in CLEAN.items() if k != "annual_income"}
    application["age"] = None
    result = validate_application({"application": application})
    assert result == {"missing_fields": ["age", "annual_income"], "is_complete": False}


def test_validate_does_not_mutate_state():
    state = {"application": copy.deepcopy(CLEAN)}
    validate_application(state)
    assert state == {"application": CLEAN}


# ── The checks ─────────────────────────────────────────────────────────────


def test_affordability():
    assert check_affordability({"application": CLEAN}) == {"risk_flags": []}
    assert check_affordability({"application": RISKY}) == {"risk_flags": ["high_loan_to_income"]}


def test_affordability_flags_zero_income_instead_of_crashing():
    assert check_affordability({"application": dict(CLEAN, annual_income=0)}) == {
        "risk_flags": ["high_loan_to_income"]
    }


def test_age():
    assert check_age({"application": dict(CLEAN, age=18)}) == {"risk_flags": []}
    assert check_age({"application": dict(CLEAN, age=17)}) == {"risk_flags": ["under_18"]}


def test_age_zero_is_under_18():
    assert check_age({"application": dict(CLEAN, age=0)}) == {"risk_flags": ["under_18"]}


def test_employment():
    def flags(status):
        return check_employment({"application": dict(CLEAN, employment_status=status)})

    assert flags("employed") == {"risk_flags": []}
    assert flags("unemployed") == {"risk_flags": ["no_employment_income"]}
    assert flags("self_employed") == {"risk_flags": ["self_employed_needs_review"]}


def test_sanctions():
    assert check_sanctions({"application": CLEAN}) == {"risk_flags": []}
    assert check_sanctions({"application": dict(CLEAN, applicant_name="  Ivan PETROV ")}) == {
        "risk_flags": ["sanctions_match"]
    }


def test_sanctions_tool_and_screening_use_the_same_lookup():
    for name in ("Ivan Petrov", "Priya Shah"):
        assert check_sanctions_list.invoke({"applicant_name": name}) == {
            "on_sanctions_list": is_sanctioned(name)
        }
    assert is_sanctioned(None) is False


def test_combine_is_only_a_join_point():
    assert combine({"risk_flags": ["under_18"]}) == {}


# ── Parallel checks in the pipeline ────────────────────────────────────────


def test_risk_flags_reducer_appends():
    hint = get_type_hints(PipelineState, include_extras=True)["risk_flags"]
    reducer = hint.__metadata__[0]
    assert reducer(["a"], ["b"]) == ["a", "b"], "parallel checks must append, not overwrite"


def test_pipeline_collects_flags_from_all_parallel_checks(fake_structured_model):
    details = ApplicationDetails(
        applicant_name="Ivan Petrov",
        age=45,
        annual_income=20_000,
        loan_amount=15_000,
        employment_status="self_employed",
    )
    graph = build_graph(fake_structured_model(details))
    text = "Ivan Petrov, 45, self-employed, £20,000 a year, want £15,000"
    final = graph.invoke({"messages": [HumanMessage(text)]})
    assert sorted(final["risk_flags"]) == [
        "high_loan_to_income",
        "sanctions_match",
        "self_employed_needs_review",
    ]
