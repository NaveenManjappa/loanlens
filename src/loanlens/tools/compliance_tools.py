"""The compliance agent's tools. Each docstring is the description the model reads."""

from langchain_core.tools import tool

from loanlens.integrations.compliance_registry import is_sanctioned, vulnerability_markers


@tool
def check_sanctions_list(applicant_name: str) -> dict:
    """Screen the applicant against the financial sanctions list. Use for EVERY applicant."""
    return {"on_sanctions_list": is_sanctioned(applicant_name)}


@tool
def check_vulnerability_markers(applicant_name: str) -> dict:
    """Check whether the applicant has any recorded vulnerability markers that need
    extra care under Consumer Duty (e.g. bereavement, health, financial difficulty)."""
    return {"markers": vulnerability_markers(applicant_name)}


COMPLIANCE_TOOLS = [check_sanctions_list, check_vulnerability_markers]
