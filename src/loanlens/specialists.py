"""Lesson 10: specialist agents as nodes in the pipeline (multi-agent, "custom workflow" pattern).

    ... refer_to_underwriter ─┬─> risk_specialist ───────┬─> underwriter_review (⏸)
                              └─> compliance_specialist ─┘
                                  (in parallel)

Each specialist is a whole create_agent graph (Lessons 6/10) called INSIDE a node function:
  - it gets its OWN private conversation (one request built from the application),
    so its tool calls and results never touch the applicant's `messages`
  - it returns ONE thing to the pipeline: a short report under specialist_reports[name]

YOUR JOB:  uv run pytest tests/test_lesson10_multi_agent.py -v  -> green
  1. merge_reports (a custom reducer)   2. make_specialist_node   3. default_specialists
  + pipeline.py and underwriting.py changes listed at the top of the test file
"""

from langchain_core.messages import ToolMessage  # noqa: F401

from loanlens.investigator_v2 import apply_policy, make_request  # noqa: F401


def merge_reports(left: dict | None, right: dict | None) -> dict:
    """Reducer for specialist_reports. Two specialists write in the SAME super-step (Lesson 2),
    each returning {"<its name>": report}. Merge them: {**left, **right}. Treat None as {}."""
    return {**(left or {}), **(right or {})}


def first_line(messages) -> str:
    """The specialist's verdict: first line of its final message ("" if empty)."""
    content = str(messages[-1].content).strip()
    return content.splitlines()[0].strip() if content else ""


def make_specialist_node(name: str, agent, verdict=first_line):
    """Return a node function that:
      1. builds a PRIVATE input: {"messages": [make_request(state["application"])]}
      2. runs the agent:  result = agent.invoke(private_input, {"recursion_limit": 100})
      3. returns ONLY  {"specialist_reports": {name: {
             "verdict":    verdict(result["messages"]),
             "memo":       the final message's content,
             "tools_used": the names of every ToolMessage in result["messages"], in order,
         }}}
    Never return "messages": the applicant's conversation must not see the agent's working."""

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
    """The real specialists, built from the real model:
        {"risk": (build_agent(model), apply_policy),          # Lesson 6 investigator + policy in code
         "compliance": (build_compliance_agent(model), first_line)}
    Import build_agent / build_compliance_agent inside the function."""
    from loanlens.investigator_v2 import build_agent  # avoid circular import
    from loanlens.compliance import (
        build_compliance_agent,
    )  # avoid circular import

    return {
        "risk": (build_agent(model), apply_policy),
        "compliance": (build_compliance_agent(model), first_line),
    }
