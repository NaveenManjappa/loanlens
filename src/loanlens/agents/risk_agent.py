"""The risk specialist agent.

The model never runs a tool itself: it asks for one, the agent runs it and returns a ToolMessage.
Its tools (tools.risk_tools) fake external systems (a credit bureau, HMRC income data).

Middleware supplies the guardrails: PII redaction (also of tool results), retries for transient tool
failures, a readable error for the model when retries run out, and a cap on model calls.
risk_verdict then applies policy in code to the agent's finished conversation.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from loanlens.agents.middleware import pii_middleware
from loanlens.agents.prompts import load_prompt
from loanlens.tools.risk_tools import RISK_TOOLS

MAX_MODEL_CALLS = 6  # up to 5 tool rounds + a final answer
TRANSIENT_ERRORS = (ConnectionError, TimeoutError)  # worth retrying: the network, not our code
REFER = "REFER TO SENIOR UNDERWRITER"  # must match the verdict named in prompts/risk_agent.md
SYSTEM_PROMPT = load_prompt("risk_agent.md")


def on_tool_error(exc: Exception, request) -> str | None:
    """Turn a transient tool failure into a message for the model; return None to let anything else raise.

    The message names the exception type, never its text, which could leak internal details.
    """
    if isinstance(exc, TRANSIENT_ERRORS):
        tool_name = request.tool_call["name"]
        return f"`{tool_name}` is unavailable right now ({type(exc).__name__}). Do not guess its data."
    return None


def build_risk_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    retry_initial_delay: float = 1.0,
) -> CompiledStateGraph:
    """Build the risk agent (tools default to RISK_TOOLS).

    Middleware order matters: retries sit inside error handling, so only the final failure reaches
    on_tool_error.
    """
    return create_agent(
        model=model,
        tools=tools if tools is not None else RISK_TOOLS,
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


def risk_verdict(messages: list[BaseMessage]) -> str:
    """The risk agent's final recommendation, with policy applied in code.

    Never recommend approval without a credit check: if lookup_credit_file failed, refer.
    Otherwise use the first line of the agent's answer.
    """
    for message in messages:
        credit_check_failed = (
            isinstance(message, ToolMessage)
            and message.name == "lookup_credit_file"
            and message.status == "error"
        )
        if credit_check_failed:
            return REFER

    content = str(messages[-1].content).strip()
    if not content:
        return REFER
    return content.splitlines()[0].strip()
