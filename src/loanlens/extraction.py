"""Extraction: a free-text application in, typed ApplicationDetails out.

The only LLM call on the main path. The model returns structured output (a Pydantic object, not
prose), and every field is optional: anything the applicant didn't state stays None, and validation
asks for it instead of the model guessing.
"""

from collections.abc import Callable
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from loanlens.state import PipelineState

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


EXTRACTION_PROMPT = """You extract loan application details from an applicant's message for a UK lender.

Rules:
- All amounts are in GBP.
- annual_income is per YEAR. If the applicant gives a monthly figure, multiply it by 12.
- Do NOT make up any information. If the applicant doesn't say it, return null. Never make a guess.
- Set employment_status only when it is stated or clearly implied (“employed”, “salary”, “work for”, “self-employed”, “freelance”, “retired”). Otherwise leave it empty. A job title alone (“nurse”, “plumber”) is NOT an employment status: it stays null and LoanLens asks.
"""


def make_extract_node(model: BaseChatModel) -> Callable[[PipelineState], dict]:
    """Build the extract_details node around a model (passed in, so tests can use a fake)."""
    structured_model = model.with_structured_output(ApplicationDetails)

    def extract_details(state: PipelineState) -> dict:
        """Ask the model for ApplicationDetails and write them to state['application']."""
        details = structured_model.invoke([SystemMessage(EXTRACTION_PROMPT), *state["messages"]])
        return {"application": details.model_dump()}

    return extract_details
