"""Lesson 11 stretch: the age guard. Turn an evaluator into a guardrail.

The EVALUATOR age_not_invented measures how often the model invents an age (offline, on a dataset).
The GUARD ground_age makes sure an invented age never reaches screening (at runtime, every case):

    extract_details -> ground_age -> validate_application

If the extracted age isn't in anything the APPLICANT wrote, set it to None. validate_application
then sees a missing age and LoanLens asks for it: the Lesson 4 safety net catches it.

One rule (age_in_text), two users (the evaluator and the guard), so they can never disagree.

YOUR JOB:  uv run pytest tests/test_lesson11b_age_guard.py -v  -> green
  1. age_in_text   2. applicant_text   3. ground_age
  4. evals.py:     age_not_invented's body becomes one call to age_in_text
  5. pipeline.py:  extract_details -> ground_age -> validate_application
"""

import re  # noqa: F401

from langchain_core.messages import HumanMessage  # noqa: F401


def age_in_text(age: int | None, text: str) -> bool:
    """True if age is None (nothing to check), or if it appears in text as a STANDALONE number.
    Standalone = not part of a bigger number or an amount of money:
        "I'm 34, a nurse"  -> found          "aged 134"         -> not found
        "Age: 34."         -> found          "earning £34,000"  -> not found
                                             "on 34.5k"         -> not found
    Hint: r"\\b34\\b" is NOT enough (in "34,000" the comma counts as a word boundary).
    Use lookarounds: "not preceded by £, a digit, '.' or ','" and "not followed by
    an optional '.' or ',' and then a digit":
        rf"(?<![£\\d.,]){age}(?![.,]?\\d)" """
    if age is None:
        return True
    return re.search(rf"(?<![£\d.,]){age}(?![.,]?\d)", text) is not None


def applicant_text(messages: list) -> str:
    """Join the content of the HumanMessages only (newline-separated).
    LoanLens's own AIMessages ("you must be 18 or over") are not evidence of anything."""
    return "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))


def ground_age(state) -> dict:
    """Node. Reads state["application"] and state["messages"].
    Age grounded (or None) -> return {}            (an empty update: change nothing)
    Age NOT grounded       -> return {"application": {...same application..., "age": None}}"""
    age = state["application"].get("age")
    text = applicant_text(state["messages"])
    if age_in_text(age, text):
        return {}
    return {"application": {**state["application"], "age": None}}

