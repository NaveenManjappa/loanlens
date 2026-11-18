"""The lending policy as plain functions, without a graph."""

from loanlens.services.policy import (
    affordability_flags,
    age_flags,
    debt_to_income,
    employment_flags,
    missing_fields,
)

COMPLETE = {
    "applicant_name": "Priya Shah",
    "age": 34,
    "annual_income": 42_000,
    "loan_amount": 8_000,
    "employment_status": "employed",
}


def test_complete_application_has_no_missing_fields_and_no_flags():
    assert missing_fields(COMPLETE) == []
    assert affordability_flags(COMPLETE) == []
    assert age_flags(COMPLETE) == []
    assert employment_flags(COMPLETE) == []


def test_missing_fields_keeps_the_policy_order():
    assert missing_fields({"loan_amount": 1, "applicant_name": None}) == [
        "applicant_name",
        "age",
        "annual_income",
        "employment_status",
    ]


def test_loan_of_exactly_half_income_is_not_flagged_but_more_is():
    assert affordability_flags({**COMPLETE, "loan_amount": 21_000}) == []
    assert affordability_flags({**COMPLETE, "loan_amount": 21_001}) == ["high_loan_to_income"]


def test_debt_to_income_rejects_a_zero_term():
    assert "error" in debt_to_income(42_000, 0, 8_000, term_months=0)
