"""Lesson 5: the tools the risk investigator agent can call.

A tool is a plain function + a name + a description + typed arguments.
The MODEL never runs a tool. It only *asks* for one (a "tool call"); your
graph runs it and sends the result back as a ToolMessage.

The docstring IS the description the model reads when deciding what to call,
so write it for the model: what it does, when to use it, what comes back.

Two tools are done (they fake external systems with canned data).
YOU write calculate_affordability: plain arithmetic, exposed as a tool.
"""

from langchain_core.tools import tool

MAX_DEBT_TO_INCOME = 0.4  # LoanLens policy v0: monthly debts incl. new loan <= 40% of gross monthly income

# ── Fake external systems (in real life: a credit bureau API, HMRC/bank data) ─

_CREDIT_FILES = {
    "priya shah": {"credit_score": 870, "missed_payments_12m": 0, "existing_monthly_debt": 150},
    "sam jones": {
        "credit_score": 640,
        "missed_payments_12m": 2,
        "existing_monthly_debt": 450,
        # Real bureau files carry bank details. The model must never see these (Lesson 7).
        "bank_details": {"sort_code": "20-45-77", "account_number": "41234567"},
    },
    "tom reid": {"credit_score": 910, "missed_payments_12m": 0, "existing_monthly_debt": 0},
}
_VERIFIED_SELF_EMPLOYED_INCOME = {
    "sam jones": {"tax_year_2024_25": 31_000, "tax_year_2025_26": 36_500},
    "tom reid": {"tax_year_2024_25": 58_000, "tax_year_2025_26": 61_000},
}


@tool
def lookup_credit_file(applicant_name: str) -> dict:
    """Look up the applicant's credit file at the credit bureau.

    Returns credit_score (0-999, higher is better), missed_payments_12m, and
    existing_monthly_debt in GBP. Use this for every investigation."""
    record = _CREDIT_FILES.get(applicant_name.strip().lower())
    return record or {"error": f"No credit file found for '{applicant_name}'"}


@tool
def verify_self_employed_income(applicant_name: str) -> dict:
    """Fetch verified annual income (GBP) from the last two tax years for a
    SELF-EMPLOYED applicant. Only useful when employment_status is self_employed."""
    record = _VERIFIED_SELF_EMPLOYED_INCOME.get(applicant_name.strip().lower())
    return record or {"error": f"No verified self-employed income for '{applicant_name}'"}


@tool
def calculate_affordability(
    annual_income: float, existing_monthly_debt: float, loan_amount: float, term_months: int = 36
) -> dict:
    """Check whether the new loan is affordable under LoanLens policy.

    Args:
        annual_income: gross annual income in GBP
        existing_monthly_debt: current monthly debt repayments in GBP
        loan_amount: total amount to borrow in GBP
        term_months: repayment term in months (default 36)

    Returns monthly_repayment, debt_to_income, and affordable (bool)."""
    
    #   monthly_repayment = loan_amount / term_months              (round to 2 dp; no interest in policy v0)
    #   debt_to_income    = (existing_monthly_debt + monthly_repayment) / (annual_income / 12)   (round to 2 dp)
    #   affordable        = debt_to_income <= MAX_DEBT_TO_INCOME
    # return {"monthly_repayment": ..., "debt_to_income": ..., "affordable": ...}
    monthly_repayment = round(loan_amount / term_months, 2)
    debt_to_income = round((monthly_repayment+existing_monthly_debt)/(annual_income/12),2)
    affordable = debt_to_income <= MAX_DEBT_TO_INCOME
    return {
        "monthly_repayment":monthly_repayment,
        "debt_to_income":debt_to_income,
        "affordable":affordable
    }

TOOLS = [lookup_credit_file, verify_self_employed_income, calculate_affordability]
