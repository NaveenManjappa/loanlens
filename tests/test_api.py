"""The HTTP API and content-filter recovery (ADR 0002). Fake models only: free and offline."""

import openai
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from loanlens.api import UNAVAILABLE, create_app
from loanlens.cases import (
    BLOCKED_REPLY,
    WITHHELD,
    continue_application,
    is_content_filter_error,
)
from loanlens.extraction import ApplicationDetails
from loanlens.pipeline import build_graph

PRIYA = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
)
PRIYA_TEXT = "Priya Shah, 34, employed nurse, £42,000 a year, want £8,000"
SAM = ApplicationDetails(
    applicant_name="Sam Jones", age=29, annual_income=40_000, loan_amount=5_000, employment_status="self_employed"
)
SAM_TEXT = "Sam Jones, 29, self-employed, about £40,000 a year, want £5,000"


class FakeContentFilterError(Exception):
    """What Azure's 400 looks like to our code: an exception whose .code is "content_filter"."""

    code = "content_filter"


class FakeModel:
    """A structured-output model that behaves like Azure: it refuses any prompt containing
    BADWORD, can be told to fail with any exception, and otherwise returns scripted answers."""

    def __init__(self, answers, fail_with=None):
        self.answers, self.fail_with = list(answers), fail_with

    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages, *args, **kwargs):
        if self.fail_with is not None:
            raise self.fail_with
        if "BADWORD" in " ".join(str(m.content) for m in messages):
            raise FakeContentFilterError("The response was filtered")
        return self.answers.pop(0)


def graph_with(*answers, fail_with=None):
    return build_graph(FakeModel(answers, fail_with), checkpointer=InMemorySaver())


# ── Part A: content filter (cases.py) ──────────────────────────────────────


def test_is_content_filter_error():
    assert is_content_filter_error(FakeContentFilterError()) is True
    assert is_content_filter_error(openai.ContentFilterFinishReasonError()) is True, "a filtered COMPLETION"
    assert is_content_filter_error(ValueError("nope")) is False
    assert is_content_filter_error(RuntimeError("content_filter in the text is not enough")) is False


def test_blocked_message_gets_a_calm_reply_not_a_crash():
    graph = graph_with(PRIYA)
    final = continue_application(graph, "case-1", "you BADWORD")
    assert final["outcome"] == "awaiting_information"
    assert isinstance(final["messages"][-1], AIMessage) and final["messages"][-1].content == BLOCKED_REPLY


def test_blocked_text_is_withheld_from_the_case():
    graph = graph_with(PRIYA)
    final = continue_application(graph, "case-1", "you BADWORD")
    contents = [m.content for m in final["messages"]]
    assert WITHHELD in contents
    assert not any("BADWORD" in c for c in contents), "the blocked text must not stay in the checkpoint"


def test_case_continues_after_a_blocked_message():
    graph = graph_with(PRIYA)
    continue_application(graph, "case-1", "you BADWORD")
    final = continue_application(graph, "case-1", PRIYA_TEXT)
    assert final["outcome"] == "approved_in_principle", "a blocked message must not block the case forever"


def test_other_errors_are_not_swallowed():
    graph = graph_with(fail_with=RuntimeError("database is down"))
    with pytest.raises(RuntimeError):
        continue_application(graph, "case-1", PRIYA_TEXT)


# ── Part B: the API (api.py) ───────────────────────────────────────────────


def client_for(graph):
    return TestClient(create_app(graph))


def test_health():
    assert client_for(graph_with()).get("/health").json() == {"status": "ok"}


def test_send_message_returns_the_case_view():
    response = client_for(graph_with(PRIYA)).post("/cases/case-1/messages", json={"text": PRIYA_TEXT})
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"] == "case-1"
    assert body["outcome"] == "approved_in_principle"
    assert body["reply"].startswith("Good news")
    assert body["awaiting_underwriter"] is False


def test_incomplete_application_asks_for_details():
    no_age = PRIYA.model_copy(update={"age": None})
    body = client_for(graph_with(no_age)).post("/cases/c2/messages", json={"text": "Priya, employed"}).json()
    assert body["outcome"] == "awaiting_information"
    assert "age" in body["reply"]


def test_get_case():
    client = client_for(graph_with(PRIYA))
    client.post("/cases/case-1/messages", json={"text": PRIYA_TEXT})
    assert client.get("/cases/case-1").json()["outcome"] == "approved_in_principle"


def test_unknown_case_is_404():
    assert client_for(graph_with()).get("/cases/nobody").status_code == 404


def test_invalid_input_is_422():
    client = client_for(graph_with(PRIYA))
    assert client.post("/cases/case-1/messages", json={"text": ""}).status_code == 422
    assert client.post("/cases/bad%20id!/messages", json={"text": "hi"}).status_code == 422


def test_blocked_message_through_the_api_is_200():
    body = client_for(graph_with(PRIYA)).post("/cases/case-1/messages", json={"text": "you BADWORD"}).json()
    assert body["reply"] == BLOCKED_REPLY


def test_referral_and_underwriter_decision():
    client = client_for(graph_with(SAM))
    body = client.post("/cases/sam/messages", json={"text": SAM_TEXT}).json()
    assert body["outcome"] == "referred" and body["awaiting_underwriter"] is True

    review = client.get("/cases/sam/review").json()
    assert review["application"]["applicant_name"] == "Sam Jones"

    decision = {"decision": "approve", "reason": "Two years of accounts seen", "underwriter_id": "uw-42"}
    body = client.post("/cases/sam/decision", json=decision).json()
    assert body["outcome"] == "approved_in_principle"
    assert body["awaiting_underwriter"] is False


def test_review_and_decision_need_a_pending_review():
    client = client_for(graph_with(PRIYA))
    client.post("/cases/case-1/messages", json={"text": PRIYA_TEXT})
    assert client.get("/cases/case-1/review").status_code == 404
    decision = {"decision": "approve", "reason": "ok", "underwriter_id": "uw-42"}
    assert client.post("/cases/case-1/decision", json=decision).status_code == 409


def test_decision_must_be_complete():
    client = client_for(graph_with(SAM))
    client.post("/cases/sam/messages", json={"text": SAM_TEXT})
    assert client.post("/cases/sam/decision", json={"decision": "approve", "reason": "", "underwriter_id": "uw"}).status_code == 422
    assert client.post("/cases/sam/decision", json={"decision": "maybe", "reason": "x", "underwriter_id": "uw"}).status_code == 422


def test_model_outage_is_503_and_leaks_nothing():
    outage = openai.APIConnectionError(message="connection refused for NI AB123456C", request=None)
    response = client_for(graph_with(fail_with=outage)).post("/cases/case-1/messages", json={"text": PRIYA_TEXT})
    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE}, "never send internal error text to the customer"
