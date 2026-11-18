"""The compliance specialist agent.

It doesn't judge creditworthiness: it checks rules that apply regardless, such as sanctions and
Consumer Duty vulnerability markers.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from loanlens.agents.middleware import pii_middleware
from loanlens.agents.prompts import load_prompt
from loanlens.tools.compliance_tools import COMPLIANCE_TOOLS

MAX_MODEL_CALLS = 4
COMPLIANCE_PROMPT = load_prompt("compliance_agent.md")


def build_compliance_agent(model: BaseChatModel) -> CompiledStateGraph:
    """The compliance agent: its tools, PII redaction and a cap on model calls."""
    return create_agent(
        model,
        COMPLIANCE_TOOLS,
        system_prompt=COMPLIANCE_PROMPT,
        middleware=[
            *pii_middleware(),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ],
    )
