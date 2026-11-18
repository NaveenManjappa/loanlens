"""Validation and policy screening nodes. The rules themselves are in services.policy.

    validate_application -> incomplete: ask the applicant
                         -> complete:   check_affordability | check_age | check_employment | check_sanctions
                                        (in parallel) -> combine

Each check appends its flags to risk_flags. Sanctions screening runs for every complete application,
and a match is a flag, so it always reaches a human.
"""

from loanlens.graph.state import PipelineState
from loanlens.integrations.compliance_registry import is_sanctioned
from loanlens.services.policy import (
    affordability_flags,
    age_flags,
    employment_flags,
    missing_fields,
)


def validate_application(state: PipelineState) -> dict:
    """List the required fields that are absent or None."""
    missing = missing_fields(state["application"])
    return {"missing_fields": missing, "is_complete": not missing}


def check_affordability(state: PipelineState) -> dict:
    return {"risk_flags": affordability_flags(state["application"])}


def check_age(state: PipelineState) -> dict:
    return {"risk_flags": age_flags(state["application"])}


def check_employment(state: PipelineState) -> dict:
    return {"risk_flags": employment_flags(state["application"])}


def check_sanctions(state: PipelineState) -> dict:
    """Flag sanctions_match if the applicant's name is on the sanctions list."""
    if is_sanctioned(state["application"]["applicant_name"]):
        return {"risk_flags": ["sanctions_match"]}
    return {"risk_flags": []}


def combine(state: PipelineState) -> dict:
    """Join point: runs once every check has finished, so routing sees all the flags."""
    return {}
