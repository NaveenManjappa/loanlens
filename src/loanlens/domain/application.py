"""The loan application: what LoanLens extracts from an applicant's messages."""

from typing import Literal

from pydantic import BaseModel, Field

EmploymentStatus = Literal["employed", "student", "self_employed", "unemployed", "retired"]


class ApplicationDetails(BaseModel):
    """Loan application details extracted from an applicant's message."""

    # Every field defaults to None: if the applicant didn't say it, we want None, not a guess.
    applicant_name: str | None = Field(default=None, description="Applicant's full name")
    age: int | None = Field(default=None, description="Applicant's age in years")
    loan_amount: float | None = Field(
        default=None, description="Amount the applicant wants to borrow, in GBP"
    )
    annual_income: float | None = Field(default=None, description="Annual income per year in GBP")
    employment_status: EmploymentStatus | None = Field(
        default=None, description="Employment status of the applicant"
    )
