"""LoanLens lending policy (illustrative). Every rule and threshold traces back to this file.

Plain functions over the extracted application (a dict in the ApplicationDetails shape). The
screening nodes in graph.screening and the risk agent's affordability tool both call these, so
the policy is applied the same way wherever it is used.
"""

from typing import Any

REQUIRED_FIELDS = ["applicant_name", "age", "annual_income", "loan_amount", "employment_status"]
MIN_AGE = 18
MAX_LOAN_TO_INCOME = 0.5  # flag a loan of MORE than 50% of annual income
MAX_DEBT_TO_INCOME = 0.4  # monthly debts incl. the new loan <= 40% of gross monthly income
DEFAULT_TERM_MONTHS = 36

Application = dict[str, Any]


def missing_fields(application: Application) -> list[str]:
    """The required fields that are absent or None, in REQUIRED_FIELDS order."""
    return [field for field in REQUIRED_FIELDS if application.get(field) is None]


def affordability_flags(application: Application) -> list[str]:
    """high_loan_to_income if the loan is over MAX_LOAN_TO_INCOME of income (or income is not positive)."""
    loan_amount = application["loan_amount"]
    annual_income = application["annual_income"]
    if annual_income <= 0 or loan_amount / annual_income > MAX_LOAN_TO_INCOME:
        return ["high_loan_to_income"]
    return []


def age_flags(application: Application) -> list[str]:
    """under_18 if the applicant is younger than MIN_AGE."""
    if application["age"] < MIN_AGE:
        return ["under_18"]
    return []


def employment_flags(application: Application) -> list[str]:
    """Unemployed and self-employed applicants need review."""
    employment_status = application["employment_status"]
    if employment_status == "unemployed":
        return ["no_employment_income"]
    if employment_status == "self_employed":
        return ["self_employed_needs_review"]
    return []


def debt_to_income(
    annual_income: float,
    existing_monthly_debt: float,
    loan_amount: float,
    term_months: int = DEFAULT_TERM_MONTHS,
) -> dict:
    """Monthly repayment, debt-to-income ratio and whether the loan is affordable.

    Returns {"error": ...} for a non-positive income or term (the agent reads errors as data).
    """
    if annual_income <= 0 or term_months <= 0:
        return {"error": "annual_income and term_months must be positive"}
    monthly_repayment = round(loan_amount / term_months, 2)  # no interest in policy v0
    monthly_income = annual_income / 12
    ratio = round((existing_monthly_debt + monthly_repayment) / monthly_income, 2)
    return {
        "monthly_repayment": monthly_repayment,
        "debt_to_income": ratio,
        "affordable": ratio <= MAX_DEBT_TO_INCOME,
    }
