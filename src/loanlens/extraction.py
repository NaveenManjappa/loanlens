"""Lesson 3: the first LLM node. Free-text application in, typed data out.

    START -> extract_details -> END

The applicant's message arrives as a HumanMessage in state["messages"].
The node asks the model for STRUCTURED OUTPUT (an ApplicationDetails object,
not prose) and writes it to state["application"], the same shape your
Lesson 1 and Lesson 2 graphs already consume.

YOUR JOB:  uv run pytest tests/test_lesson03_extraction.py -v   -> green
Then the real thing:  uv run python scripts/lesson03_demo.py
"""

from typing import Literal  # noqa: F401  (you'll need it for employment_status)

from langchain_core.messages import SystemMessage  # noqa: F401
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: F401
from pydantic import BaseModel, Field


class ApplicationDetails(BaseModel):
    """Loan application details extracted from an applicant's message."""

    # Every field is Optional (`| None`): if the applicant didn't say it, we want
    # None, NOT a guess. Lesson 1's validate_application turns None into "missing".
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
    """Return the node function. The model is passed IN (dependency injection),
    so tests can hand it a fake model and never call Azure."""

    def extract_details(state: ExtractionState) -> dict:
        """1. Wrap the model:  model.with_structured_output(ApplicationDetails)
        2. Invoke it with [SystemMessage(EXTRACTION_PROMPT), *state["messages"]]
        3. Return {"application": <the result as a plain dict>}  (hint: .model_dump())"""
        structured = model.with_structured_output(ApplicationDetails)
        details = structured.invoke(
            [SystemMessage(EXTRACTION_PROMPT), *state["messages"]]
        )
        return {"application": details.model_dump()}

    return extract_details


def build_graph(model=None):
    """START -> extract_details -> END.  If no model is given, use the real one:
    `from loanlens.llm import get_model` (import it INSIDE this function, so tests
    with a fake model don't need your .env)."""
    if not model:
        from loanlens.llm import get_model

        model = get_model()
    builder = StateGraph(ExtractionState)
    builder.add_node(make_extract_node(model))
    builder.add_edge(START, "extract_details")
    builder.add_edge("extract_details", END)
    return builder.compile()
