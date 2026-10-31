"""Lesson 2 spec. Run: uv run pytest tests/test_lesson02_screening.py -v"""

from typing import get_type_hints

from langchain_core.messages import AIMessage, HumanMessage

from loanlens.screening import (
    ScreeningState,
    build_graph,
    check_affordability,
    check_age,
    check_employment,
    combine,
)

CLEAN = {
    "applicant_name": "Priya Shah",
    "age": 34,
    "annual_income": 50_000,
    "loan_amount": 25_000,  # exactly 0.5 -> NOT over the limit
    "employment_status": "employed",
}
RISKY = dict(CLEAN, age=17, loan_amount=40_000, employment_status="self_employed")


# ── The checks (plain functions) ───────────────────────────────────────────


def test_affordability():
    assert check_affordability({"application": CLEAN}) == {"risk_flags": []}
    assert check_affordability({"application": RISKY}) == {"risk_flags": ["high_loan_to_income"]}


def test_age():
    assert check_age({"application": dict(CLEAN, age=18)}) == {"risk_flags": []}
    assert check_age({"application": dict(CLEAN, age=17)}) == {"risk_flags": ["under_18"]}


def test_employment():
    def flags(status):
        return check_employment({"application": dict(CLEAN, employment_status=status)})

    assert flags("employed") == {"risk_flags": []}
    assert flags("unemployed") == {"risk_flags": ["no_employment_income"]}
    assert flags("self_employed") == {"risk_flags": ["self_employed_needs_review"]}


def test_combine_refer():
    out = combine({"risk_flags": ["under_18", "high_loan_to_income"]})
    assert out["decision_hint"] == "refer"
    assert len(out["messages"]) == 1, "return only the NEW message"
    text = out["messages"][0].content
    assert "refer" in text and "high_loan_to_income, under_18" in text


def test_combine_clear():
    out = combine({"risk_flags": []})
    assert out["decision_hint"] == "clear"
    assert "none" in out["messages"][0].content


# ── The state schema ───────────────────────────────────────────────────────


def test_risk_flags_has_a_reducer():
    hint = get_type_hints(ScreeningState, include_extras=True)["risk_flags"]
    reducer = getattr(hint, "__metadata__", None)
    assert reducer, (
        "risk_flags has no reducer, so parallel writes will clash. "
        "Use Annotated[list[str], <reducer>]."
    )
    assert reducer[0](["a"], ["b"]) == ["a", "b"], "the reducer should APPEND lists"


# ── The graph ──────────────────────────────────────────────────────────────


def test_checks_fan_out_from_start():
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    for check in ("check_affordability", "check_age", "check_employment"):
        assert ("__start__", check) in edges, f"START should go straight to {check}"
        assert (check, "combine") in edges, f"{check} should go to combine"


def test_graph_collects_flags_from_all_parallel_checks():
    final = build_graph().invoke({"application": RISKY})
    assert sorted(final["risk_flags"]) == [
        "high_loan_to_income",
        "self_employed_needs_review",
        "under_18",
    ]
    assert final["decision_hint"] == "refer"


def test_graph_clean_application():
    final = build_graph().invoke({"application": CLEAN})
    assert final["risk_flags"] == []
    assert final["decision_hint"] == "clear"


def test_messages_are_appended_not_overwritten():
    note = HumanMessage("Applicant note: I'm starting a new job next month.")
    final = build_graph().invoke({"application": CLEAN, "messages": [note]})
    assert len(final["messages"]) == 2
    assert final["messages"][0].content == note.content  # the original survived
    assert isinstance(final["messages"][-1], AIMessage)
