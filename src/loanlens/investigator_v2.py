"""Lesson 6: the same risk investigator, rebuilt with LangChain's create_agent.

Lesson 5 (by hand)                    Lesson 6 (prebuilt)
───────────────────────────────────   ───────────────────────────────────────────────
investigator node + bind_tools        create_agent(model, tools, system_prompt=...)
your tool_node                        built-in tool node (also handles unknown tools / bad args)
should_continue (natural stop)        built in: stops when the model makes no tool calls
tool_rounds hard cap + escalate       ModelCallLimitMiddleware(run_limit=..., exit_behavior="end")
(nothing)                             ToolRetryMiddleware: quiet retries for flaky APIs
(nothing)                             ToolErrorMiddleware: final failure -> error the model can read

create_agent returns a compiled LangGraph graph: same .invoke / .stream / .get_graph().

YOUR JOB:  uv run pytest tests/test_lesson06_create_agent.py -v  -> green
  1. on_tool_error   2. build_agent   3. apply_policy
"""

import json

from langchain.agents import create_agent  # noqa: F401
from langchain.agents.middleware import (  # noqa: F401
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from loanlens.tools import TOOLS
from loanlens.pii import pii_middleware
MAX_MODEL_CALLS = 6  # hard cap: 6 model calls = up to 5 tool rounds + a final answer (same budget as Lesson 5)
TRANSIENT_ERRORS = (
    ConnectionError,
    TimeoutError,
)  # worth retrying: the network, not our code

SYSTEM_PROMPT = """You are a loan risk investigator at a UK lender.
Investigate the referred application with your tools, then reply with a short recommendation.
The FIRST line must be exactly one of: APPROVE, DECLINE, REFER TO SENIOR UNDERWRITER.
Then 2-4 bullet points of evidence. Only use facts returned by your tools or present in the
application. If a tool reports it is unavailable, say so; never guess the missing data.
Call independent tools together in one step."""


def make_request(application: dict) -> HumanMessage:
    """The application travels in the first human message (the system prompt stays fixed)."""
    return HumanMessage(
        "Investigate this referred application:\n" + json.dumps(application)
    )


def on_tool_error(exc: Exception, request) -> str | None:
    """Called by ToolErrorMiddleware when a tool STILL fails after retries.

    - TRANSIENT_ERRORS (outage, timeout) -> return a message for the model, e.g.
      "`lookup_credit_file` is unavailable right now (ConnectionError). Do not guess its data."
      (the tool name is request.tool_call["name"]; name the exception TYPE, never the raw
       exception text: it can leak internal details)
    - anything else (a bug: ValueError, KeyError...) -> return None, so it propagates and the
      run fails LOUDLY. Don't hide your own bugs from yourself."""
    if isinstance(exc, TRANSIENT_ERRORS):
        return f"`{request.tool_call['name']}` is unavailable right now ({type(exc).__name__}). Do not guess its data."
    return None


def build_agent(model=None, tools=TOOLS, retry_initial_delay: float = 1.0):
    """create_agent(model, tools, system_prompt=SYSTEM_PROMPT, middleware=[...]) with, IN THIS ORDER:
      1. ToolErrorMiddleware(on_error=on_tool_error)
      2. ToolRetryMiddleware(max_retries=2, retry_on=TRANSIENT_ERRORS, on_failure="error",
                             initial_delay=retry_initial_delay)
      3. ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end")
    Order matters: retry must sit INSIDE (after) error handling, so retries happen first and
    only the final failure reaches on_tool_error. (Tests pass retry_initial_delay=0 so they're fast.)
    If model is None: from loanlens.llm import get_model."""
    if model is None:
        from loanlens.llm import get_model

        model = get_model()
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            *pii_middleware(),
            ToolErrorMiddleware(on_error=on_tool_error),
            ToolRetryMiddleware(
                max_retries=2,
                retry_on=TRANSIENT_ERRORS,
                on_failure="error",
                initial_delay=retry_initial_delay,
            ),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ],
    )


def apply_policy(messages: list[BaseMessage]) -> str:
    """Policy in CODE, applied to the agent's finished conversation. Return the final recommendation:

    - If any lookup_credit_file ToolMessage has status == "error" -> "REFER TO SENIOR UNDERWRITER"
      (no credit check = never auto-APPROVE, whatever the model wrote)
    - otherwise -> the FIRST line of the last message's content, stripped.

    Hint: ToolMessages carry .name (the tool) and .status ("success" or "error")."""
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "lookup_credit_file" and m.status == "error":
            return "REFER TO SENIOR UNDERWRITER"
    last_message = messages[-1]
    return last_message.content.splitlines()[0].strip() if last_message.content else "REFER TO SENIOR UNDERWRITER"