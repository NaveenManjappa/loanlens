"""A risk-investigator agent loop built by hand in LangGraph (reference implementation).

    START -> investigator --should_continue--+--> tools --> investigator   (the loop)
                                             +--> END                      (natural stop)
                                             +--> escalate --> END         (hard cap)

The pipeline uses investigator_v2 (create_agent + middleware); this version shows the same loop
without the prebuilt agent.
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from loanlens.tools import TOOLS

TOOLS_BY_NAME = {t.name: t for t in TOOLS}
MAX_TOOL_ROUNDS = 5  # hard cap on tool rounds

INVESTIGATOR_PROMPT = """You are a loan risk investigator at a UK lender.
The application below was referred for manual review. Investigate it with your tools, then
reply with a short recommendation: APPROVE, DECLINE or REFER TO SENIOR UNDERWRITER,
followed by 2-4 bullet points of evidence. Only use facts returned by your tools or present
in the application. If a tool returns an error, say so; never guess the missing data.

Application:
{application}"""


class InvestigationState(MessagesState, total=False):
    application: dict
    tool_rounds: int  # how many times the tools node has run


def make_investigator_node(model):
    """Build the agent node: the model with tools bound, prompted with the application."""

    def investigator(state: InvestigationState) -> dict:
        model_with_tools = model.bind_tools(TOOLS)
        response = model_with_tools.invoke(
            [
                SystemMessage(
                    INVESTIGATOR_PROMPT.format(
                        application=json.dumps(state["application"])
                    )
                ),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    return investigator


def tool_node(state: InvestigationState) -> dict:
    """Run every tool call in the last AIMessage; an unknown tool returns an error the model can read."""
    tool_rounds = state.get("tool_rounds", 0)
    tool_rounds += 1
    last_message = state["messages"][-1]
    messages = []

    for tool_call in last_message.tool_calls:
        name = tool_call.get("name")
        args = tool_call.get("args")
        tool_call_id = tool_call.get("id")
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            messages.append(
                ToolMessage(
                    content=f"Error: unknown tool '{name}'. Available tools: {', '.join(TOOLS_BY_NAME)}",
                    tool_call_id=tool_call_id,
                )
            )
        else:
            result = tool.invoke(args)
            messages.append(
                ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)
            )
    return {"messages": messages, "tool_rounds": tool_rounds}


def should_continue(state: InvestigationState) -> Literal["tools", "escalate", END]:
    """Stopping condition: the hard cap first, then the natural stop when no tools are requested."""
    tool_rounds = state.get("tool_rounds", 0)
    last_message = state["messages"][-1]
    if tool_rounds >= MAX_TOOL_ROUNDS:
        return "escalate"
    elif not last_message.tool_calls:
        return END
    else:
        return "tools"


def escalate(state: InvestigationState) -> dict:
    return {
        "messages": [
            AIMessage(
                f"Investigation stopped after {MAX_TOOL_ROUNDS} tool rounds without a conclusion. "
                "REFER TO SENIOR UNDERWRITER."
            )
        ]
    }


def build_graph(model=None):
    """Compile the agent loop."""
    if model is None:
        from loanlens.llm import get_model

        model = get_model()
    builder = StateGraph(InvestigationState)
    builder.add_node(make_investigator_node(model))
    builder.add_node("tools", tool_node)
    builder.add_node(escalate)

    builder.add_edge(START, "investigator")
    builder.add_conditional_edges(
        "investigator", should_continue, ["tools", "escalate", END]
    )
    builder.add_edge("tools", "investigator")
    builder.add_edge("escalate", END)
    return builder.compile()
