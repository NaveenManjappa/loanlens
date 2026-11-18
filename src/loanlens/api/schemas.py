"""Request and response models. Pydantic validates input, so a bad request gets a 422 before any
graph runs."""

from typing import Literal

from pydantic import BaseModel, Field


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class DecisionIn(BaseModel):
    decision: Literal["approve", "decline"]
    reason: str = Field(min_length=1)
    underwriter_id: str = Field(min_length=1)


class CaseView(BaseModel):
    """What callers see of a case: never the raw state (least exposure)."""

    case_id: str
    outcome: str | None
    reply: str | None
    awaiting_underwriter: bool
