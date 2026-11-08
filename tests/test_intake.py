"""Intake: validation, ratios and the standalone intake graph."""

import copy

from loanlens.intake import build_graph, compute_ratios, summarise, validate_application

GOOD_APP = {
    "applicant_name": "Priya Shah",
    "age": 34,
    "annual_income": 50_000,
    "loan_amount": 25_000,
    "employment_status": "employed",
}
MISSING_INCOME_APP = {k: v for k, v in GOOD_APP.items() if k != "annual_income"}


# ── Nodes are just functions: test them without any graph ─────────────────


def test_validate_complete_application():
    assert validate_application({"application": GOOD_APP}) == {
        "missing_fields": [],
        "is_complete": True,
    }


def test_validate_flags_missing_and_none_fields():
    app = dict(MISSING_INCOME_APP, age=None)
    result = validate_application({"application": app})
    assert result == {"missing_fields": ["age", "annual_income"], "is_complete": False}


def test_validate_returns_only_its_own_keys():
    """A node returns a PARTIAL update — never the whole state."""
    result = validate_application({"application": GOOD_APP})
    assert "application" not in result


def test_validate_does_not_mutate_state():
    state = {"application": copy.deepcopy(GOOD_APP)}
    validate_application(state)
    assert state == {"application": GOOD_APP}


def test_compute_loan_to_income():
    app = dict(GOOD_APP, loan_amount=20_000, annual_income=30_000)
    assert compute_ratios({"application": app}) == {"loan_to_income": 0.67}


def test_compute_handles_missing_income():
    assert compute_ratios({"application": MISSING_INCOME_APP}) == {"loan_to_income": None}


def test_summarise():
    assert summarise({"is_complete": True}) == {"status": "ready_for_review"}
    assert summarise({"is_complete": False}) == {"status": "incomplete"}


# ── The graph: wiring + state flowing through it ───────────────────────────


def test_graph_has_the_three_nodes():
    node_names = set(build_graph().get_graph().nodes)
    assert {"validate_application", "compute_ratios", "summarise"} <= node_names


def test_graph_end_to_end_complete():
    final = build_graph().invoke({"application": GOOD_APP})
    assert final["status"] == "ready_for_review"
    assert final["loan_to_income"] == 0.5
    assert final["missing_fields"] == []
    assert final["application"] == GOOD_APP  # untouched keys survive


def test_graph_end_to_end_incomplete():
    final = build_graph().invoke({"application": MISSING_INCOME_APP})
    assert final["status"] == "incomplete"
    assert final["missing_fields"] == ["annual_income"]
    assert final["loan_to_income"] is None
