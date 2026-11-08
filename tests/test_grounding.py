"""The age guard: the shared age_in_text rule, the ground_age node and its place in the pipeline."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from loanlens.cases import continue_application
from loanlens.evals import age_not_invented
from loanlens.extraction import ApplicationDetails
from loanlens.grounding import age_in_text, applicant_text, ground_age
from loanlens.pipeline import build_graph

# ── 1. age_in_text: the one rule ───────────────────────────────────────────


def test_age_in_text_finds_a_standalone_number():
    assert age_in_text(34, "I'm 34, a nurse") is True
    assert age_in_text(34, "Age: 34.") is True
    assert age_in_text(None, "anything at all") is True, "no age extracted = nothing invented"


def test_age_in_text_rejects_numbers_that_are_not_there():
    assert age_in_text(30, "I'm 34") is False
    assert age_in_text(34, "aged 134") is False, "part of a bigger number"


def test_age_in_text_rejects_money():
    # \b alone is NOT enough: in "34,000" the comma is a word boundary, so r"\b34\b" matches.
    assert age_in_text(34, "earning £34,000 a year") is False
    assert age_in_text(34, "about 34,000 a year") is False
    assert age_in_text(34, "on 34.5k") is False


# ── 2. applicant_text: only what the APPLICANT wrote ───────────────────────


def test_applicant_text_keeps_human_messages_only():
    text = applicant_text(
        [HumanMessage("Tom, plumber"), AIMessage("How old are you? You must be 18 or over."), HumanMessage("I'm 40")]
    )
    assert "Tom, plumber" in text and "I'm 40" in text
    assert "18" not in text, "LoanLens's own words don't count as evidence"


# ── 3. ground_age: the node ────────────────────────────────────────────────


def test_guard_discards_an_invented_age_and_keeps_everything_else():
    state = {
        "messages": [HumanMessage("Jordan, teacher, graduated last summer")],
        "application": {"applicant_name": "Jordan Lee", "age": 22},
    }
    assert ground_age(state) == {"application": {"applicant_name": "Jordan Lee", "age": None}}


def test_guard_changes_nothing_when_the_age_is_grounded():
    state = {"messages": [HumanMessage("Priya, 34, nurse")], "application": {"applicant_name": "Priya", "age": 34}}
    assert ground_age(state) == {}, "no change -> return an empty update"


def test_guard_ignores_numbers_loanlens_wrote():
    state = {
        "messages": [HumanMessage("Tom, plumber"), AIMessage("Applicants must be 18 or over. How old are you?")],
        "application": {"age": 18},
    }
    assert ground_age(state)["application"]["age"] is None


def test_evaluator_and_guard_share_one_rule():
    assert age_not_invented({"text": "earning £34,000 a year"}, {"application": {"age": 34}}) is False


# ── 4. In the pipeline ─────────────────────────────────────────────────────


def test_guard_runs_between_extraction_and_validation(scripted_structured_model):
    priya = ApplicationDetails(
        applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
    )
    graph = build_graph(scripted_structured_model([priya]), checkpointer=InMemorySaver())
    order = []
    for step in graph.stream(
        {"messages": [("user", "Priya Shah, 34, nurse, £42,000 a year, want £8,000")]},
        {"configurable": {"thread_id": "p"}},
        stream_mode="updates",
    ):
        order.extend(step)
    assert order.index("extract_details") < order.index("ground_age") < order.index("validate_application")


def test_invented_age_makes_loanlens_ask(scripted_structured_model):
    invented = ApplicationDetails(
        applicant_name="Jordan Lee", age=22, annual_income=31_000, loan_amount=3_000, employment_status="employed"
    )
    graph = build_graph(scripted_structured_model([invented]), checkpointer=InMemorySaver())
    final = continue_application(
        graph, "case-jordan", "Jordan Lee, employed teacher, graduated last summer, £31,000 salary, want £3,000"
    )
    assert final["outcome"] == "awaiting_information"
    assert "age" in final["missing_fields"]


def test_age_given_in_a_later_reply_is_accepted(scripted_structured_model):
    no_age = ApplicationDetails(
        applicant_name="Tom Reid", age=None, annual_income=60_000, loan_amount=8_000, employment_status="employed"
    )
    with_age = no_age.model_copy(update={"age": 40})
    graph = build_graph(scripted_structured_model([no_age, with_age]), checkpointer=InMemorySaver())
    first = continue_application(graph, "case-tom", "Tom Reid, employed plumber, £60k a year, want £8,000")
    assert first["outcome"] == "awaiting_information"
    second = continue_application(graph, "case-tom", "I'm 40")
    assert second["outcome"] == "approved_in_principle", "the guard reads the WHOLE conversation"
