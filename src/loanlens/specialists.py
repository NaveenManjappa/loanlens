"""Specialist agents as nodes in the pipeline.

    refer_to_underwriter -> risk_specialist + compliance_specialist (in parallel) -> underwriter_review

Each specialist is a whole create_agent graph, called inside a node with a private conversation built
from the redacted application, so its tool calls never touch the applicant's messages. It returns one
short report under specialist_reports[name]; state.merge_reports combines the parallel writes.
"""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from loanlens.compliance import build_compliance_agent
from loanlens.risk import build_risk_agent, risk_verdict
from loanlens.state import PipelineState

AGENT_RECURSION_LIMIT = 100

# A verdict function reads an agent's finished conversation and returns its one-line verdict.
Verdict = Callable[[list[BaseMessage]], str]
# Specialist name -> (agent, verdict function). The name becomes the node "<name>_specialist".
Specialists = dict[str, tuple[CompiledStateGraph, Verdict]]


def make_request(application: dict[str, Any]) -> HumanMessage:
    """The application travels in the first human message (the system prompt stays fixed)."""
    return HumanMessage("Investigate this referred application:\n" + json.dumps(application))


def first_line(messages: list[BaseMessage]) -> str:
    """The default verdict: the first line of the agent's final message ("" if empty)."""
    content = str(messages[-1].content).strip()
    return content.splitlines()[0].strip() if content else ""


def make_specialist_node(
    name: str, agent: CompiledStateGraph, verdict: Verdict = first_line
) -> Callable[[PipelineState], dict]:
    """Build a node that runs the agent privately and returns only its report."""

    def specialist_node(state: PipelineState) -> dict:
        private_input = {"messages": [make_request(state["application"])]}
        result = agent.invoke(private_input, {"recursion_limit": AGENT_RECURSION_LIMIT})
        messages = result["messages"]
        tools_used = [message.name for message in messages if isinstance(message, ToolMessage)]
        report = {
            "verdict": verdict(messages),
            "memo": str(messages[-1].content),
            "tools_used": tools_used,
        }
        return {"specialist_reports": {name: report}}

    return specialist_node


def default_specialists(model: BaseChatModel) -> Specialists:
    """The real specialists: the risk agent (with policy in code) and the compliance agent."""
    return {
        "risk": (build_risk_agent(model), risk_verdict),
        "compliance": (build_compliance_agent(model), first_line),
    }
