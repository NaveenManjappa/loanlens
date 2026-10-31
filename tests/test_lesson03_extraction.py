"""Lesson 3 spec.

Offline (fake model, free):   uv run pytest tests/test_lesson03_extraction.py -v
Live (real Foundry model):    LOANLENS_LIVE=1 uv run pytest tests/test_lesson03_extraction.py -v -m live
"""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from loanlens.extraction import (
    EXTRACTION_PROMPT,
    ApplicationDetails,
    build_graph,
    make_extract_node,
)

FIELDS = {"applicant_name", "age", "annual_income", "loan_amount", "employment_status"}
EMAIL = HumanMessage("Hi, I'm Priya Shah, 34, employed as a nurse on £3,500 a month. I'd like to borrow £12,000.")
EXPECTED = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=12_000, employment_status="employed"
)


# ── The schema ─────────────────────────────────────────────────────────────


def test_schema_has_exactly_the_five_fields():
    assert set(ApplicationDetails.model_fields) == FIELDS


def test_every_field_defaults_to_none():
    """Missing information must become None, never a made-up value."""
    assert ApplicationDetails().model_dump() == {f: None for f in FIELDS}


def test_employment_status_only_accepts_known_values():
    ApplicationDetails(employment_status="self_employed")  # fine
    with pytest.raises(ValidationError):
        ApplicationDetails(employment_status="contractor")


def test_annual_income_description_says_per_year():
    desc = (ApplicationDetails.model_fields["annual_income"].description or "").lower()
    assert "year" in desc or "annual" in desc, "the model reads descriptions: say it's per year"


def test_prompt_has_a_no_guessing_rule():
    assert "TODO" not in EXTRACTION_PROMPT, "write your no-invention rule into the prompt"


# ── The node (with a fake model) ───────────────────────────────────────────


def test_node_asks_for_structured_output(fake_structured_model):
    fake = fake_structured_model(EXPECTED)
    make_extract_node(fake)({"messages": [EMAIL]})
    assert fake.schema is ApplicationDetails


def test_node_sends_system_prompt_then_conversation(fake_structured_model):
    fake = fake_structured_model(EXPECTED)
    make_extract_node(fake)({"messages": [EMAIL]})
    assert isinstance(fake.sent[0], SystemMessage)
    assert fake.sent[0].content == EXTRACTION_PROMPT
    assert fake.sent[1].content == EMAIL.content


def test_node_returns_application_as_plain_dict(fake_structured_model):
    out = make_extract_node(fake_structured_model(EXPECTED))({"messages": [EMAIL]})
    assert out == {"application": EXPECTED.model_dump()}
    assert type(out["application"]) is dict


# ── The graph (with a fake model) ──────────────────────────────────────────


def test_graph_end_to_end_with_fake_model(fake_structured_model):
    final = build_graph(fake_structured_model(EXPECTED)).invoke({"messages": [EMAIL]})
    assert final["application"]["annual_income"] == 42_000
    assert final["messages"][0].content == EMAIL.content


# ── LIVE: the real model (skipped unless LOANLENS_LIVE=1) ─────────────────


@pytest.mark.live
def test_live_converts_monthly_income_to_annual():
    app = build_graph().invoke({"messages": [EMAIL]})["application"]
    assert app["annual_income"] == 42_000
    assert app["employment_status"] == "employed"


@pytest.mark.live
def test_live_does_not_invent_missing_age():
    msg = HumanMessage("Tom Reid here. Self-employed plumber, about £60k a year. Looking for £8,000.")
    app = build_graph().invoke({"messages": [msg]})["application"]
    assert app["age"] is None, f"model invented an age: {app['age']}"
    assert app["employment_status"] == "self_employed"

