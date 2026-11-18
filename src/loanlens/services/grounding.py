"""The age-grounding rule, shared by the runtime guard and the evaluator.

graph.intake.ground_age discards an extracted age that fails this rule; evals.age_not_invented
scores runs with it. Both apply exactly the same rule because both call age_in_text.
"""

import re


def age_in_text(age: int | None, text: str) -> bool:
    """True if age is None, or appears in text as a standalone number.

    Standalone means not part of a bigger number or an amount of money: "I'm 34" matches, but
    "aged 134", "£34,000" and "34.5k" don't (a plain word-boundary match would accept "34,000").
    """
    if age is None:
        return True
    return re.search(rf"(?<![£\d.,]){age}(?![.,]?\d)", text) is not None
