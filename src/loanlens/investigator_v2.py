"""The risk-investigator agent, built with LangChain's create_agent (the risk specialist).

Middleware supplies the guardrails: ToolRetryMiddleware retries transient tool failures,
ToolErrorMiddleware turns a final failure into a message the model can read, and
ModelCallLimitMiddleware caps the number of model calls. apply_policy then applies policy in code
to the agent's finished conversation.
"""

import json

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from loanlens.tools import TOOLS
from loanlens.pii import pii_middleware
MAX_MODEL_CALLS = 6  # hard cap: 6 model calls = up to 5 tool rounds + a final answer
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
    """Turn a transient tool failure into a message for the model; let anything else propagate.

    The message names the exception type, never its text, which could leak internal details.
    """
    if isinstance(exc, TRANSIENT_ERRORS):
        return f"`{request.tool_call['name']}` is unavailable right now ({type(exc).__name__}). Do not guess its data."
    return None


def build_agent(model=None, tools=TOOLS, retry_initial_delay: float = 1.0):
    """Build the agent.

    Middleware order matters: retries sit inside error handling, so only the final failure reaches
    on_tool_error.
    """
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
    """The final recommendation, in code.

    Refer to a senior underwriter if the credit check failed (never auto-approve without one);
    otherwise use the first line of the agent's answer.
    """
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "lookup_credit_file" and m.status == "error":
            return "REFER TO SENIOR UNDERWRITER"
    last_message = messages[-1]
    return last_message.content.splitlines()[0].strip() if last_message.content else "REFER TO SENIOR UNDERWRITER"