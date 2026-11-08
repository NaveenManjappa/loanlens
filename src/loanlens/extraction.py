"""Extraction: a free-text application in, typed ApplicationDetails out.

The only LLM call on the main path. The model returns structured output (a Pydantic object, not
prose), and every field is optional: anything the applicant didn't state stays None, and validation
asks for it instead of the model guessing.
"""

from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field


class ApplicationDetails(BaseModel):
    """Loan application details extracted from an applicant's message."""

    # Every field is Optional (`| None`): if the applicant didn't say it, we want
    # None, NOT a guess. validate_application turns None into "missing".
    applicant_name: str | None = Field(
        default=None, description="Applicant's full name"
    )
    age: int | None = Field(default=None, description="Applicant's age in years")
    loan_amount: float | None = Field(
        default=None,
        description="Amount the applicant wants to borrow, in GBP",
    )
    annual_income: float | None = Field(
        default=None, description="Annual income per year in GBP"
    )
    employment_status: (
        Literal["employed", "student", "self_employed", "unemployed", "retired"] | None
    ) = Field(default=None, description="Employment status of the applicant")


EXTRACTION_PROMPT = """You extract loan application details from an applicant's message for a UK lender.

Rules:
- All amounts are in GBP.
- annual_income is per YEAR. If the applicant gives a monthly figure, multiply it by 12.
- Do NOT make up any information. If the applicant doesn't say it, return null. Never make a guess.
"""


class ExtractionState(MessagesState, total=False):
    application: dict  # written by extract_details


def make_extract_node(model):
    """Build the extract_details node around a model (injected, so tests can use a fake)."""

    def extract_details(state: ExtractionState) -> dict:
        """Ask the model for ApplicationDetails and write it to state['application']."""
        structured = model.with_structured_output(ApplicationDetails)
        details = structured.invoke(
            [SystemMessage(EXTRACTION_PROMPT), *state["messages"]]
        )
        return {"application": details.model_dump()}

    return extract_details


def build_graph(model=None):
    """A standalone extraction graph (START -> extract_details -> END), used in tests."""
    if not model:
        from loanlens.llm import get_model

        model = get_model()
    builder = StateGraph(ExtractionState)
    builder.add_node(make_extract_node(model))
    builder.add_edge(START, "extract_details")
    builder.add_edge("extract_details", END)
    return builder.compile()
