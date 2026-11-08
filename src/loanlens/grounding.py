"""The age guard: an evaluator promoted to a runtime guardrail.

    extract_details -> ground_age -> validate_application

If the extracted age doesn't appear in anything the applicant wrote, it's discarded and validation
asks for it. age_in_text is shared with the age_not_invented evaluator, so the eval and the guard
apply exactly the same rule.
"""

import re

from langchain_core.messages import HumanMessage


def age_in_text(age: int | None, text: str) -> bool:
    """True if age is None, or appears in text as a standalone number.

    Standalone means not part of a bigger number or an amount of money: "I'm 34" matches, but
    "aged 134", "£34,000" and "34.5k" don't (a plain word-boundary match would accept "34,000").
    """
    if age is None:
        return True
    return re.search(rf"(?<![£\d.,]){age}(?![.,]?\d)", text) is not None


def applicant_text(messages: list) -> str:
    """The text of the applicant's own messages; LoanLens's replies are not evidence."""
    return "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))


def ground_age(state) -> dict:
    """Node: discard the extracted age if the applicant never wrote it (otherwise an empty update)."""
    age = state["application"].get("age")
    text = applicant_text(state["messages"])
    if age_in_text(age, text):
        return {}
    return {"application": {**state["application"], "age": None}}

