"""Validation and policy screening. Plain code: deterministic and auditable.

    validate_application -> incomplete: ask the applicant
                         -> complete:   check_affordability | check_age | check_employment | check_sanctions
                                        (in parallel) -> combine

Each check appends its flags to risk_flags. Sanctions screening runs for every complete application,
and a match is a flag, so it always reaches a human.
"""

from loanlens.compliance import is_sanctioned
from loanlens.state import PipelineState

# LoanLens policy (illustrative): every rule traces back to here.
REQUIRED_FIELDS = ["applicant_name", "age", "annual_income", "loan_amount", "employment_status"]
MIN_AGE = 18
MAX_LOAN_TO_INCOME = 0.5  # flag a loan of MORE than 50% of annual income


def validate_application(state: PipelineState) -> dict:
    """List the required fields that are absent or None, in REQUIRED_FIELDS order."""
    application = state["application"]
    missing = [field for field in REQUIRED_FIELDS if application.get(field) is None]
    return {"missing_fields": missing, "is_complete": not missing}


def check_affordability(state: PipelineState) -> dict:
    """Flag high_loan_to_income if the loan is over MAX_LOAN_TO_INCOME of income (or income is not positive)."""
    loan_amount = state["application"]["loan_amount"]
    annual_income = state["application"]["annual_income"]
    if annual_income <= 0 or loan_amount / annual_income > MAX_LOAN_TO_INCOME:
        return {"risk_flags": ["high_loan_to_income"]}
    return {"risk_flags": []}


def check_age(state: PipelineState) -> dict:
    """Flag under_18 if the applicant is younger than MIN_AGE."""
    if state["application"]["age"] < MIN_AGE:
        return {"risk_flags": ["under_18"]}
    return {"risk_flags": []}


def check_employment(state: PipelineState) -> dict:
    """Flag unemployed and self-employed applicants for review."""
    employment_status = state["application"]["employment_status"]
    if employment_status == "unemployed":
        return {"risk_flags": ["no_employment_income"]}
    if employment_status == "self_employed":
        return {"risk_flags": ["self_employed_needs_review"]}
    return {"risk_flags": []}


def check_sanctions(state: PipelineState) -> dict:
    """Flag sanctions_match if the applicant's name is on the sanctions list."""
    if is_sanctioned(state["application"]["applicant_name"]):
        return {"risk_flags": ["sanctions_match"]}
    return {"risk_flags": []}


def combine(state: PipelineState) -> dict:
    """Join point: runs once every check has finished, so routing sees all the flags."""
    return {}
