"""Lesson 5: the risk investigator, LoanLens's first true AGENT, built by hand.

            ┌──────────────────────────────┐
            v                              │
  START -> investigator --(should_continue)┼--> tools ──┘     (model asked for tools)
                                           ├--> END            (natural stop: no tool calls)
                                           └--> escalate -> END (hard cap: too many rounds)

The model decides which tools to call, in what order, and when it's done.
YOU decide the guard rails: which tools exist, and the hard cap.

YOUR JOB:  uv run pytest tests/test_lesson05_investigator.py -v  -> green
  1. calculate_affordability (in tools.py)   2. make_investigator_node
  3. tool_node                               4. should_continue
  5. build_graph
"""

import json  # noqa: F401  (tool results are dicts; ToolMessage content must be a string)
from typing import Literal  # noqa: F401

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage  # noqa: F401
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401

from loanlens.tools import TOOLS

TOOLS_BY_NAME = {t.name: t for t in TOOLS}
MAX_TOOL_ROUNDS = 5  # the HARD CAP from your Lesson 0 stopping condition

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
    """Return the node. Inside it:
    1. bind the tools:    model.bind_tools(TOOLS)
    2. invoke it with     [SystemMessage(INVESTIGATOR_PROMPT.format(application=...)), *state["messages"]]
                          (json.dumps(state["application"]) makes a tidy application string)
    3. return {"messages": [response]}   (the AIMessage, possibly carrying .tool_calls)"""

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
    """Run EVERY tool call in the last AIMessage (there can be several).
    For each call:  tool = TOOLS_BY_NAME[call["name"]];  result = tool.invoke(call["args"])
                    -> ToolMessage(content=json.dumps(result), tool_call_id=call["id"])
    If the name isn't a known tool, DON'T crash: return a ToolMessage whose content
    starts with "Error: unknown tool" so the model can recover.
    Also return "tool_rounds": previous value (default 0) + 1."""
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
    """The stopping condition, in code:
    - hard cap:     tool_rounds >= MAX_TOOL_ROUNDS            -> "escalate"
    - natural stop: last message has no tool_calls            -> END
    - otherwise:    the model wants tools                     -> "tools"
    (Check the hard cap FIRST. Why? Answer that when you report back.)"""
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
    """Nodes: "investigator" (make_investigator_node(model)), "tools" (tool_node), "escalate".
    Edges: START -> investigator;  investigator --should_continue--> tools / escalate / END
           tools -> investigator   (<- THIS edge is what makes it a loop)
           escalate -> END
    If model is None: from loanlens.llm import get_model."""
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
    # builder.add_edge("investigator", END)
    return builder.compile()
