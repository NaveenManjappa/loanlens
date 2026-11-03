"""Lesson 6 spec.

Offline:  uv run pytest tests/test_lesson06_create_agent.py -v
Live:     LOANLENS_LIVE=1 uv run pytest tests/test_lesson06_create_agent.py -v -m live
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from loanlens.investigator_v2 import (
    MAX_MODEL_CALLS,
    apply_policy,
    build_agent,
    make_request,
    on_tool_error,
)
from loanlens.tools import TOOLS

SAM = {
    "applicant_name": "Sam Jones",
    "age": 29,
    "annual_income": 33_600,
    "loan_amount": 20_000,
    "employment_status": "self_employed",
}


def ai_call(name, args, call_id="call_0"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


LOOKUP_SAM = ai_call("lookup_credit_file", {"applicant_name": "Sam Jones"})


class _Request:  # the bit of ToolCallRequest that on_tool_error uses
    tool_call = {"name": "lookup_credit_file", "args": {}, "id": "call_0"}


def make_failing_bureau(exc_type):
    """A lookup_credit_file that always raises exc_type, and counts how often it was called."""
    attempts = {"n": 0}

    @tool
    def lookup_credit_file(applicant_name: str) -> dict:
        """Look up the applicant's credit file at the credit bureau."""
        attempts["n"] += 1
        raise exc_type("internal detail: bureau host 10.0.0.7 refused connection")

    return lookup_credit_file, attempts


def run(agent):
    return agent.invoke({"messages": [make_request(SAM)]})["messages"]


# ── on_tool_error ──────────────────────────────────────────────────────────


def test_on_tool_error_explains_outages_to_the_model():
    for exc in (ConnectionError("x"), TimeoutError("x")):
        text = on_tool_error(exc, _Request())
        assert "lookup_credit_file" in text and "unavailable" in text
        assert type(exc).__name__ in text


def test_on_tool_error_never_leaks_raw_exception_text():
    text = on_tool_error(ConnectionError("bureau host 10.0.0.7 refused connection"), _Request())
    assert "10.0.0.7" not in text


def test_on_tool_error_lets_bugs_propagate():
    assert on_tool_error(ValueError("bug"), _Request()) is None
    assert on_tool_error(KeyError("bug"), _Request()) is None


# ── build_agent: the happy path ────────────────────────────────────────────


def test_agent_is_a_langgraph_graph_with_your_tools(scripted_model):
    model = scripted_model([AIMessage("REFER TO SENIOR UNDERWRITER\n- x")])
    agent = build_agent(model)
    assert {"model", "tools"} <= set(agent.get_graph().nodes)
    run(agent)
    assert set(model.bound_tools) == {t.name for t in TOOLS}
    assert model.calls[0][0].type == "system"


def test_agent_loops_then_stops_naturally(scripted_model):
    model = scripted_model([LOOKUP_SAM, AIMessage("REFER TO SENIOR UNDERWRITER\n- 2 missed payments")])
    messages = run(build_agent(model))
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai"]
    assert '"missed_payments_12m": 2' in messages[2].content


# ── Middleware: retries, errors, limits ────────────────────────────────────


def test_outage_is_retried_then_reported_to_the_model(scripted_model):
    bureau, attempts = make_failing_bureau(ConnectionError)
    model = scripted_model([LOOKUP_SAM, AIMessage("REFER TO SENIOR UNDERWRITER\n- bureau unavailable")])
    messages = run(build_agent(model, tools=[bureau], retry_initial_delay=0))
    assert attempts["n"] == 3, "1 call + 2 retries"
    tool_msg = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_msg.status == "error"
    assert "unavailable" in tool_msg.content
    assert messages[-1].type == "ai", "the run should finish, not crash"


def test_bugs_are_not_retried_and_fail_loudly(scripted_model):
    bureau, attempts = make_failing_bureau(ValueError)
    model = scripted_model([LOOKUP_SAM, AIMessage("done")])
    with pytest.raises(ValueError):
        run(build_agent(model, tools=[bureau], retry_initial_delay=0))
    assert attempts["n"] == 1, "a bug isn't transient: no retries"


def test_stubborn_agent_hits_the_model_call_limit(scripted_model):
    model = scripted_model([LOOKUP_SAM])  # asks for the same tool forever
    messages = run(build_agent(model))
    assert len(model.calls) == MAX_MODEL_CALLS
    assert "limit" in messages[-1].content.lower()


# ── apply_policy: the rule the model can't overrule ────────────────────────


def _conversation(lookup_status, final_text):
    return [
        HumanMessage("Investigate"),
        LOOKUP_SAM,
        ToolMessage("...", tool_call_id="call_0", name="lookup_credit_file", status=lookup_status),
        AIMessage(final_text),
    ]


def test_policy_keeps_the_models_recommendation_when_checks_ran():
    msgs = _conversation("success", "  DECLINE \n- DTI 0.62")
    assert apply_policy(msgs) == "DECLINE"


def test_policy_overrules_approve_when_credit_check_failed():
    msgs = _conversation("error", "APPROVE\n- looks fine")
    assert apply_policy(msgs) == "REFER TO SENIOR UNDERWRITER"


def test_policy_ignores_errors_from_other_tools():
    msgs = _conversation("success", "APPROVE\n- fine")
    msgs.insert(3, ToolMessage("err", tool_call_id="call_9", name="verify_self_employed_income", status="error"))
    assert apply_policy(msgs) == "APPROVE"


# ── LIVE ───────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_agent_handles_a_bureau_outage():
    bureau, attempts = make_failing_bureau(ConnectionError)
    others = [t for t in TOOLS if t.name != "lookup_credit_file"]
    messages = build_agent(tools=[bureau, *others], retry_initial_delay=0).invoke(
        {"messages": [make_request(SAM)]}, {"recursion_limit": 100}
    )["messages"]
    assert attempts["n"] >= 3
    assert apply_policy(messages) == "REFER TO SENIOR UNDERWRITER"
