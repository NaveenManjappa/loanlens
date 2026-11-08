"""Specialist agents as nodes in the pipeline.

    refer_to_underwriter -> risk_specialist + compliance_specialist (in parallel) -> underwriter_review

Each specialist is a whole create_agent graph, called inside a node with a private conversation built
from the redacted application, so its tool calls never touch the applicant's messages. It returns one
short report under specialist_reports[name], and merge_reports combines the parallel writes.
"""

from langchain_core.messages import ToolMessage

from loanlens.investigator_v2 import apply_policy, make_request


def merge_reports(left: dict | None, right: dict | None) -> dict:
    """Reducer for specialist_reports: merge the reports written in the same super-step."""
    return {**(left or {}), **(right or {})}


def first_line(messages) -> str:
    """The specialist's verdict: first line of its final message ("" if empty)."""
    content = str(messages[-1].content).strip()
    return content.splitlines()[0].strip() if content else ""


def make_specialist_node(name: str, agent, verdict=first_line):
    """Build a node that runs the agent privately and returns only its report (verdict, memo, tools used)."""

    def node(state):
        private_input = {"messages": [make_request(state["application"])]}
        result = agent.invoke(private_input, {"recursion_limit": 100})
        final_message = result["messages"][-1]
        tools_used = [
            msg.name
            for msg in result["messages"]
            if isinstance(msg, ToolMessage)
        ]
        return {
            "specialist_reports": {
                name: {
                    "verdict": verdict(result["messages"]),
                    "memo": str(final_message.content),
                    "tools_used": tools_used,
                }
            }
        }

    return node


def default_specialists(model=None) -> dict:
    """The real specialists: the risk investigator (with policy in code) and the compliance agent."""
    from loanlens.investigator_v2 import build_agent  # avoid circular import
    from loanlens.compliance import (
        build_compliance_agent,
    )  # avoid circular import

    return {
        "risk": (build_agent(model), apply_policy),
        "compliance": (build_compliance_agent(model), first_line),
    }
