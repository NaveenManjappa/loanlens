"""The compliance specialist agent, and the sanctions lookup used by screening.

The agent doesn't judge creditworthiness: it checks rules that apply regardless, such as sanctions
and Consumer Duty vulnerability markers. Sanctions screening of every applicant is plain code
(is_sanctioned, called by screening.check_sanctions); the agent's tool repeats it for its report.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.graph.state import CompiledStateGraph

from loanlens.pii import pii_middleware

MAX_MODEL_CALLS = 4

# Fake external systems. In real life: an HM Treasury / OFSI screening service and a CRM.
_SANCTIONS_LIST = {"ivan petrov"}
_VULNERABILITY_MARKERS = {"tom reid": ["recent bereavement (self-declared)"]}

COMPLIANCE_PROMPT = """You are a compliance checker at a UK lender. You do NOT assess creditworthiness.
Check the application with your tools, then reply. The FIRST line must be exactly one of:
COMPLIANT, NEEDS REVIEW, NOT COMPLIANT
Then 1-3 bullet points of evidence. A sanctions match is always NOT COMPLIANT.
Any vulnerability marker is NEEDS REVIEW. Only use facts returned by your tools."""


def _normalise(name: str) -> str:
    return name.strip().lower()


def is_sanctioned(applicant_name: str | None) -> bool:
    """True if the name is on the sanctions list.

    Name-only matching can produce false positives, so a match must go to a human, never straight
    to approve or decline.
    """
    if not applicant_name:
        return False
    return _normalise(applicant_name) in _SANCTIONS_LIST


@tool
def check_sanctions_list(applicant_name: str) -> dict:
    """Screen the applicant against the financial sanctions list. Use for EVERY applicant."""
    return {"on_sanctions_list": is_sanctioned(applicant_name)}


@tool
def check_vulnerability_markers(applicant_name: str) -> dict:
    """Check whether the applicant has any recorded vulnerability markers that need
    extra care under Consumer Duty (e.g. bereavement, health, financial difficulty)."""
    return {"markers": _VULNERABILITY_MARKERS.get(_normalise(applicant_name), [])}


COMPLIANCE_TOOLS = [check_sanctions_list, check_vulnerability_markers]


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
