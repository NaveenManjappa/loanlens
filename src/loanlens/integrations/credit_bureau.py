"""A fake credit bureau and HMRC income service, with canned data.

In production these would be HTTP clients (with timeouts and retries). The risk agent's tools in
tools.risk_tools call these functions; nothing else does.
"""

_CREDIT_FILES = {
    "priya shah": {"credit_score": 870, "missed_payments_12m": 0, "existing_monthly_debt": 150},
    "sam jones": {
        "credit_score": 640,
        "missed_payments_12m": 2,
        "existing_monthly_debt": 450,
        # Real bureau files carry bank details. The model must never see these.
        "bank_details": {"sort_code": "20-45-77", "account_number": "41234567"},
    },
    "tom reid": {"credit_score": 910, "missed_payments_12m": 0, "existing_monthly_debt": 0},
}
_VERIFIED_SELF_EMPLOYED_INCOME = {
    "sam jones": {"tax_year_2024_25": 31_000, "tax_year_2025_26": 36_500},
    "tom reid": {"tax_year_2024_25": 58_000, "tax_year_2025_26": 61_000},
}


def _normalise(name: str) -> str:
    return name.strip().lower()


def get_credit_file(applicant_name: str) -> dict | None:
    """The applicant's credit file, or None if the bureau has no file for them."""
    return _CREDIT_FILES.get(_normalise(applicant_name))


def get_verified_self_employed_income(applicant_name: str) -> dict | None:
    """Verified income for the last two tax years, or None if there is no record."""
    return _VERIFIED_SELF_EMPLOYED_INCOME.get(_normalise(applicant_name))
