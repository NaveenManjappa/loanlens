"""Lesson 7 spec. Run: uv run pytest tests/test_lesson07_pii.py -v"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from loanlens.extraction import ApplicationDetails
from loanlens.pii import find_pii, pii_middleware, redact, redact_at_boundary, redact_pii

NINO = "AB 12 34 56 C"
EMAIL_WITH_PII = (
    f"Hi, I'm Priya Shah, 34, nurse on £42,000 a year, want £8,000. NI number {NINO}. "
    "Pay into sort code 20-45-77, account 41234567. Email priya.shah@example.co.uk"
)


def types(text):
    return [m["type"] for m in find_pii(text)]


# ── Patterns: spec → regex ─────────────────────────────────────────────────


@pytest.mark.parametrize("nino", ["AB 12 34 56 C", "AB123456C", "ce 10 20 30 d", "JZ 99 88 77 A"])
def test_valid_ninos_are_found(nino):
    assert types(f"my NI is {nino} thanks") == ["uk_nino"]


@pytest.mark.parametrize(
    "not_nino",
    [
        "QQ 12 34 56 C",  # Q is never used (HMRC uses QQ in examples precisely because it's invalid)
        "AO 12 34 56 A",  # O is never the 2nd letter
        "GB 12 34 56 A",  # banned prefix
        "ZZ 12 34 56 A",  # banned prefix
        "AB 12 34 56 E",  # suffix must be A-D
        "AB 12 34 5 C",  # only 5 digits
    ],
)
def test_invalid_ninos_are_not_found(not_nino):
    assert "uk_nino" not in types(f"ref {not_nino} end")


def test_sort_code_and_account_number():
    assert types("sort code 20-45-77, account 41234567") == ["uk_sort_code", "uk_account_number"]


def test_amounts_and_years_are_not_pii():
    assert types("I earn £42,000 a year and want £8,000 over 36 months from 2026") == []
    assert types("account 1234567 and 123456789") == [], "7 or 9 digits aren't a UK account number"


# ── redact ─────────────────────────────────────────────────────────────────


def test_redact_replaces_every_type_and_keeps_what_the_model_needs():
    out = redact(EMAIL_WITH_PII)
    for leaked in (NINO, "20-45-77", "41234567", "priya.shah@example.co.uk"):
        assert leaked not in out
    for tag in ("[REDACTED_UK_NINO]", "[REDACTED_UK_SORT_CODE]", "[REDACTED_UK_ACCOUNT_NUMBER]", "[REDACTED_EMAIL]"):
        assert tag in out
    for needed in ("Priya Shah", "34", "£42,000", "£8,000"):
        assert needed in out, "data minimisation, not data destruction"


def test_redact_leaves_clean_text_alone():
    assert redact("Tom, 41, £60k, wants £8,000") == "Tom, 41, £60k, wants £8,000"


def test_redact_at_boundary_returns_a_clean_human_message():
    msg = redact_at_boundary(EMAIL_WITH_PII)
    assert isinstance(msg, HumanMessage) and "20-45-77" not in msg.content


# ── redact_pii node ────────────────────────────────────────────────────────


def test_node_replaces_messages_by_id():
    original = HumanMessage(EMAIL_WITH_PII, id="msg-1")
    out = redact_pii({"messages": [original, AIMessage("hello", id="msg-2")]})
    assert [m.id for m in out["messages"]] == ["msg-1"], "only the message with PII, same id"
    assert NINO not in out["messages"][0].content


def test_node_returns_nothing_to_replace_when_clean():
    assert redact_pii({"messages": [HumanMessage("Tom, 41, £60k", id="m")]}) == {"messages": []}


# ── Wired into the pipeline (edit pipeline.py) ─────────────────────────────


def test_pipeline_redacts_before_the_llm_sees_anything(fake_structured_model):
    from loanlens.pipeline import build_graph

    fake = fake_structured_model(
        ApplicationDetails(
            applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
        )
    )
    graph = build_graph(fake)
    assert "redact_pii" in graph.get_graph().nodes
    final = graph.invoke({"messages": [HumanMessage(EMAIL_WITH_PII)]})
    sent_to_llm = " ".join(str(m.content) for m in fake.sent)
    assert NINO not in sent_to_llm and "41234567" not in sent_to_llm
    assert NINO not in final["messages"][0].content, "the state's copy is redacted too"
    assert final["outcome"] == "approved_in_principle"


# ── Wired into the create_agent investigator (edit investigator_v2.py) ─────


def test_pii_middleware_has_one_redactor_per_type():
    mws = pii_middleware()
    assert len(mws) == 4


def test_investigator_never_sees_bank_details_from_the_bureau(scripted_model):
    from loanlens.investigator_v2 import build_agent, make_request

    lookup = AIMessage(
        content="",
        tool_calls=[{"name": "lookup_credit_file", "args": {"applicant_name": "Sam Jones"}, "id": "call_0"}],
    )
    model = scripted_model([lookup, AIMessage("REFER TO SENIOR UNDERWRITER\n- x")])
    build_agent(model).invoke({"messages": [make_request({"applicant_name": "Sam Jones"})]})
    tool_result_seen = model.calls[1][-1].content
    assert "missed_payments_12m" in tool_result_seen, "the useful data still arrives"
    assert "20-45-77" not in tool_result_seen and "41234567" not in tool_result_seen
    assert "[REDACTED_UK_SORT_CODE]" in tool_result_seen
