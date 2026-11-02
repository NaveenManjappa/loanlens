"""Lesson 5 spec.

Offline (scripted fake model):  uv run pytest tests/test_lesson05_investigator.py -v
Live (your Foundry model):      LOANLENS_LIVE=1 uv run pytest tests/test_lesson05_investigator.py -v -m live
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END

from loanlens.investigator import (
    MAX_TOOL_ROUNDS,
    build_graph,
    make_investigator_node,
    should_continue,
    tool_node,
)
from loanlens.tools import TOOLS, calculate_affordability

SAM = {
    "applicant_name": "Sam Jones",
    "age": 29,
    "annual_income": 33_600,
    "loan_amount": 20_000,
    "employment_status": "self_employed",
}
ASK = HumanMessage("Investigate this referred application and give your recommendation.")


def ai_calls(*calls):
    """An AIMessage asking for tools. calls = (name, args) pairs."""
    return AIMessage(
        content="",
        tool_calls=[{"name": n, "args": a, "id": f"call_{i}"} for i, (n, a) in enumerate(calls)],
    )


FINAL = AIMessage("REFER TO SENIOR UNDERWRITER\n- 2 missed payments\n- debt_to_income 0.36")


# ── The tool you wrote ─────────────────────────────────────────────────────


def test_calculate_affordability_maths():
    out = calculate_affordability.invoke(
        {"annual_income": 36_000, "existing_monthly_debt": 300, "loan_amount": 9_000, "term_months": 36}
    )
    assert out == {"monthly_repayment": 250.0, "debt_to_income": 0.18, "affordable": True}


def test_calculate_affordability_over_the_limit():
    out = calculate_affordability.invoke(
        {"annual_income": 33_600, "existing_monthly_debt": 450, "loan_amount": 20_000}
    )
    assert out["monthly_repayment"] == 555.56
    assert out["debt_to_income"] == 0.36
    assert out["affordable"] is True
    out = calculate_affordability.invoke(
        {"annual_income": 24_000, "existing_monthly_debt": 450, "loan_amount": 20_000}
    )
    assert out["affordable"] is False


def test_tool_schema_is_what_the_model_sees():
    """The model only sees the name, description (docstring) and argument schema."""
    schema = calculate_affordability.args
    assert set(schema) == {"annual_income", "existing_monthly_debt", "loan_amount", "term_months"}
    assert "affordable" in calculate_affordability.description


# ── should_continue: the stopping condition in code ────────────────────────


def test_should_continue_routes_tool_calls_to_tools():
    state = {"messages": [ai_calls(("lookup_credit_file", {"applicant_name": "Sam Jones"}))], "tool_rounds": 0}
    assert should_continue(state) == "tools"


def test_should_continue_natural_stop():
    assert should_continue({"messages": [FINAL], "tool_rounds": 2}) == END


def test_should_continue_hard_cap_beats_tool_calls():
    state = {
        "messages": [ai_calls(("lookup_credit_file", {"applicant_name": "Sam Jones"}))],
        "tool_rounds": MAX_TOOL_ROUNDS,
    }
    assert should_continue(state) == "escalate"


# ── tool_node: you run the tools the model asked for ───────────────────────


def test_tool_node_runs_every_call_and_links_ids():
    msg = ai_calls(
        ("lookup_credit_file", {"applicant_name": "Sam Jones"}),
        ("verify_self_employed_income", {"applicant_name": "Sam Jones"}),
    )
    out = tool_node({"messages": [msg], "tool_rounds": 0})
    replies = out["messages"]
    assert [type(m) for m in replies] == [ToolMessage, ToolMessage]
    assert [m.tool_call_id for m in replies] == ["call_0", "call_1"]
    assert json.loads(replies[0].content)["missed_payments_12m"] == 2
    assert out["tool_rounds"] == 1


def test_tool_node_counts_rounds_from_zero():
    msg = ai_calls(("lookup_credit_file", {"applicant_name": "Priya Shah"}))
    assert tool_node({"messages": [msg]})["tool_rounds"] == 1  # no tool_rounds in state yet


def test_tool_node_survives_unknown_tool():
    msg = ai_calls(("delete_database", {}))
    reply = tool_node({"messages": [msg], "tool_rounds": 0})["messages"][0]
    assert isinstance(reply, ToolMessage)
    assert reply.content.startswith("Error: unknown tool")


# ── The investigator node ──────────────────────────────────────────────────


def test_investigator_binds_all_tools_and_sends_system_prompt(fake_tool_model):
    fake = fake_tool_model([FINAL])
    out = make_investigator_node(fake)({"application": SAM, "messages": [ASK]})
    assert {t.name for t in fake.tools} == {t.name for t in TOOLS}
    sent = fake.calls[0]
    assert isinstance(sent[0], SystemMessage) and "Sam Jones" in sent[0].content
    assert sent[1].content == ASK.content
    assert list(out) == ["messages"]
    assert out["messages"][0].content == FINAL.content


# ── The loop, end to end ───────────────────────────────────────────────────


def test_agent_loop_until_natural_stop(fake_tool_model):
    script = [
        ai_calls(("lookup_credit_file", {"applicant_name": "Sam Jones"})),
        ai_calls(
            ("calculate_affordability", {"annual_income": 33_600, "existing_monthly_debt": 450, "loan_amount": 20_000})
        ),
        FINAL,
    ]
    final = build_graph(fake_tool_model(script)).invoke({"application": SAM, "messages": [ASK]})
    kinds = [m.type for m in final["messages"]]
    assert kinds == ["human", "ai", "tool", "ai", "tool", "ai"]
    assert final["tool_rounds"] == 2
    assert final["messages"][-1].content == FINAL.content


def test_agent_that_never_stops_is_escalated(fake_tool_model):
    stubborn = fake_tool_model([ai_calls(("lookup_credit_file", {"applicant_name": "Sam Jones"}))])
    final = build_graph(stubborn).invoke({"application": SAM, "messages": [ASK]})
    assert final["tool_rounds"] == MAX_TOOL_ROUNDS
    assert final["messages"][-1].content.startswith(f"Investigation stopped after {MAX_TOOL_ROUNDS} tool rounds")


# ── LIVE ───────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_live_investigator_uses_tools_then_stops():
    final = build_graph().invoke({"application": SAM, "messages": [ASK]}, {"recursion_limit": 25})
    tool_names = [c["name"] for m in final["messages"] if m.type == "ai" for c in m.tool_calls]
    assert "lookup_credit_file" in tool_names
    assert not final["messages"][-1].tool_calls, "should end on a final answer"
