"""Specialist agents as nodes: private context, merged reports and the review payload."""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from loanlens.cases import continue_application, thread_config
from loanlens.compliance import build_compliance_agent
from loanlens.extraction import ApplicationDetails
from loanlens.investigator_v2 import apply_policy, build_agent
from loanlens.pipeline import build_graph
from loanlens.specialists import first_line, make_specialist_node, merge_reports
from loanlens.underwriting import awaiting_underwriter, submit_decision

SAM = ApplicationDetails(
    applicant_name="Sam Jones", age=29, annual_income=40_000, loan_amount=5_000, employment_status="self_employed"
)
PRIYA = ApplicationDetails(
    applicant_name="Priya Shah", age=34, annual_income=42_000, loan_amount=8_000, employment_status="employed"
)
APPROVE = {"decision": "approve", "reason": "Specialist reports reviewed", "underwriter_id": "uw-017"}


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def risk_script():
    return [
        call("lookup_credit_file", {"applicant_name": "Sam Jones"}, "r1"),
        AIMessage("REFER TO SENIOR UNDERWRITER\n- 2 missed payments in 12 months"),
    ]


def compliance_script():
    return [
        call("check_sanctions_list", {"applicant_name": "Sam Jones"}, "c1"),
        AIMessage("COMPLIANT\n- not on sanctions list"),
    ]


def specialists(scripted_model):
    risk_model, comp_model = scripted_model(risk_script()), scripted_model(compliance_script())
    return {
        "risk": (build_agent(risk_model, retry_initial_delay=0), apply_policy),
        "compliance": (build_compliance_agent(comp_model), first_line),
    }, risk_model, comp_model


# ── merge_reports ──────────────────────────────────────────────────────────


def test_merge_reports():
    assert merge_reports({"risk": 1}, {"compliance": 2}) == {"risk": 1, "compliance": 2}
    assert merge_reports(None, {"risk": 1}) == {"risk": 1}
    assert merge_reports({"risk": 1}, None) == {"risk": 1}


# ── make_specialist_node ───────────────────────────────────────────────────


def test_specialist_node_returns_only_its_report(scripted_model):
    model = scripted_model(compliance_script())
    node = make_specialist_node("compliance", build_compliance_agent(model))
    out = node({"application": SAM.model_dump(), "messages": ["the applicant's conversation"]})
    assert list(out) == ["specialist_reports"], "never return messages"
    report = out["specialist_reports"]["compliance"]
    assert report["verdict"] == "COMPLIANT"
    assert report["tools_used"] == ["check_sanctions_list"]
    assert report["memo"].startswith("COMPLIANT")


def test_specialist_gets_a_private_conversation(scripted_model):
    model = scripted_model(compliance_script())
    make_specialist_node("compliance", build_compliance_agent(model))(
        {"application": SAM.model_dump(), "messages": ["SECRET applicant chat"]}
    )
    first_call = " ".join(str(m.content) for m in model.calls[0])
    assert "Sam Jones" in first_call, "it sees the application"
    assert "SECRET applicant chat" not in first_call, "but not the applicant's conversation"


# ── In the pipeline ────────────────────────────────────────────────────────


def test_referred_case_runs_both_specialists_in_parallel_then_pauses(scripted_model, scripted_structured_model):
    specs, risk_model, comp_model = specialists(scripted_model)
    graph = build_graph(scripted_structured_model([SAM]), checkpointer=InMemorySaver(), specialists=specs)
    edges = {(e.source, e.target) for e in graph.get_graph().edges}
    assert ("refer_to_underwriter", "risk_specialist") in edges
    assert ("refer_to_underwriter", "compliance_specialist") in edges

    result = continue_application(graph, "case-sam", "Sam, 29, self-employed, £40k, want £5,000")
    assert "__interrupt__" in result
    reports = awaiting_underwriter(graph, "case-sam")["specialist_reports"]
    assert reports["risk"]["verdict"] == "REFER TO SENIOR UNDERWRITER"
    assert reports["compliance"]["verdict"] == "COMPLIANT"


def test_applicant_conversation_stays_clean(scripted_model, scripted_structured_model):
    specs, _, _ = specialists(scripted_model)
    graph = build_graph(scripted_structured_model([SAM]), checkpointer=InMemorySaver(), specialists=specs)
    result = continue_application(graph, "case-sam", "Sam, 29, self-employed, £40k, want £5,000")
    assert not any(isinstance(m, ToolMessage) for m in result["messages"]), "no agent tool chatter"
    assert not any(getattr(m, "tool_calls", None) for m in result["messages"])


def test_resuming_does_not_rerun_the_specialists(scripted_model, scripted_structured_model):
    specs, risk_model, comp_model = specialists(scripted_model)
    graph = build_graph(scripted_structured_model([SAM]), checkpointer=InMemorySaver(), specialists=specs)
    continue_application(graph, "case-sam", "Sam, 29, self-employed, £40k, want £5,000")
    calls_before = (len(risk_model.calls), len(comp_model.calls))
    final = submit_decision(graph, "case-sam", APPROVE)
    assert final["outcome"] == "approved_in_principle"
    assert (len(risk_model.calls), len(comp_model.calls)) == calls_before, "agents ran once, before the pause"


def test_clean_cases_never_call_the_specialists(scripted_model, scripted_structured_model):
    specs, risk_model, comp_model = specialists(scripted_model)
    graph = build_graph(scripted_structured_model([PRIYA]), checkpointer=InMemorySaver(), specialists=specs)
    result = continue_application(graph, "case-priya", "Priya, 34, nurse, £42k, want £8k")
    assert result["outcome"] == "approved_in_principle"
    assert risk_model.calls == [] and comp_model.calls == []


def test_reports_are_reset_on_the_next_run(scripted_model, scripted_structured_model):
    specs, _, _ = specialists(scripted_model)
    graph = build_graph(scripted_structured_model([SAM, PRIYA]), checkpointer=InMemorySaver(), specialists=specs)
    continue_application(graph, "case-x", "first application")
    submit_decision(graph, "case-x", APPROVE)
    continue_application(graph, "case-x", "a new, clean application")
    assert graph.get_state(thread_config("case-x")).values["specialist_reports"] == {}
