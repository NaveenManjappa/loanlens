"""Lesson 10: a second specialist agent, the compliance checker. (Done for you; read it.)

Same shape as Lesson 6's risk investigator: create_agent + tools + middleware.
Different job: it doesn't judge creditworthiness, it checks rules that must be followed regardless.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import tool

from loanlens.pii import pii_middleware

# ── Fake external systems ──────────────────────────────────────────────────
_SANCTIONS_LIST = {"ivan petrov"}  # in real life: an HM Treasury / OFSI screening service
_VULNERABILITY_MARKERS = {"tom reid": ["recent bereavement (self-declared)"]}


@tool
def check_sanctions_list(applicant_name: str) -> dict:
    """Screen the applicant against the financial sanctions list. Use for EVERY applicant."""
    return {"on_sanctions_list": applicant_name.strip().lower() in _SANCTIONS_LIST}


@tool
def check_vulnerability_markers(applicant_name: str) -> dict:
    """Check whether the applicant has any recorded vulnerability markers that need
    extra care under Consumer Duty (e.g. bereavement, health, financial difficulty)."""
    return {"markers": _VULNERABILITY_MARKERS.get(applicant_name.strip().lower(), [])}


COMPLIANCE_TOOLS = [check_sanctions_list, check_vulnerability_markers]

COMPLIANCE_PROMPT = """You are a compliance checker at a UK lender. You do NOT assess creditworthiness.
Check the application with your tools, then reply. The FIRST line must be exactly one of:
COMPLIANT, NEEDS REVIEW, NOT COMPLIANT
Then 1-3 bullet points of evidence. A sanctions match is always NOT COMPLIANT.
Any vulnerability marker is NEEDS REVIEW. Only use facts returned by your tools."""


def build_compliance_agent(model=None):
    if model is None:
        from loanlens.llm import get_model

        model = get_model()
    return create_agent(
        model,
        COMPLIANCE_TOOLS,
        system_prompt=COMPLIANCE_PROMPT,
        middleware=[*pii_middleware(), ModelCallLimitMiddleware(run_limit=4, exit_behavior="end")],
    )
