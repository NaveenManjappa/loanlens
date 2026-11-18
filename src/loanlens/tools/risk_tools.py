"""The risk agent's tools. Each docstring is the description the model reads.

Thin wrappers: data comes from integrations.credit_bureau and the maths from services.policy.
A tool that finds nothing returns {"error": ...} so the model reads the failure as data.
"""

from langchain_core.tools import tool

from loanlens.integrations.credit_bureau import get_credit_file, get_verified_self_employed_income
from loanlens.services.policy import DEFAULT_TERM_MONTHS, debt_to_income


@tool
def lookup_credit_file(applicant_name: str) -> dict:
    """Look up the applicant's credit file at the credit bureau.

    Returns credit_score (0-999, higher is better), missed_payments_12m, and
    existing_monthly_debt in GBP. Use this for every investigation."""
    record = get_credit_file(applicant_name)
    return record or {"error": f"No credit file found for '{applicant_name}'"}


@tool
def verify_self_employed_income(applicant_name: str) -> dict:
    """Fetch verified annual income (GBP) from the last two tax years for a
    SELF-EMPLOYED applicant. Only useful when employment_status is self_employed."""
    record = get_verified_self_employed_income(applicant_name)
    return record or {"error": f"No verified self-employed income for '{applicant_name}'"}


@tool
def calculate_affordability(
    annual_income: float,
    existing_monthly_debt: float,
    loan_amount: float,
    term_months: int = DEFAULT_TERM_MONTHS,
) -> dict:
    """Check whether the new loan is affordable under LoanLens policy.

    Args:
        annual_income: gross annual income in GBP
        existing_monthly_debt: current monthly debt repayments in GBP
        loan_amount: total amount to borrow in GBP
        term_months: repayment term in months (default 36)

    Returns monthly_repayment, debt_to_income, and affordable (bool)."""
    return debt_to_income(annual_income, existing_monthly_debt, loan_amount, term_months)


RISK_TOOLS = [lookup_credit_file, verify_self_employed_income, calculate_affordability]
