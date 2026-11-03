"""Lesson 7: PII guardrails for LoanLens. Deterministic code, never an LLM.

The rule (UK GDPR data minimisation): the model gets only the personal data it
NEEDS for the task. LoanLens needs name, age, income, loan amount, employment.
It never needs a National Insurance number, sort code, account number or email.

Three places this module is used:
  1. redact_at_boundary(text)  : at the API edge, BEFORE graph.invoke, so raw PII
                                  is never stored in a checkpoint (Lesson 8)
  2. redact_pii (graph node)   : first node of the pipeline (defence in depth)
  3. pii_middleware()          : PIIMiddleware for the create_agent investigator,
                                  scrubbing TOOL RESULTS before the model reads them

YOUR JOB:  uv run pytest tests/test_lesson07_pii.py -v  -> green
  1. NINO_PATTERN  2. SORT_CODE_PATTERN  3. ACCOUNT_NUMBER_PATTERN
  4. redact        5. redact_pii         6. pii_middleware
  then wire redact_pii into pipeline.py and pii_middleware() into investigator_v2.py
"""

import re

from langchain.agents.middleware import PIIMiddleware  # noqa: F401
from langchain_core.messages import HumanMessage

# ── Patterns ───────────────────────────────────────────────────────────────
# Write each one from the SPEC in its comment. That's the exercise: spec → regex.
# All are compiled with re.IGNORECASE by _find() below. Use \b word boundaries.

# National Insurance number (HMRC NIM39110):
#   2 letters, 6 digits, 1 suffix letter A, B, C or D.  People often type spaces: "AB 12 34 56 C".
#   - D F I Q U V are never the 1st or 2nd letter;  O is never the 2nd letter
#   - prefixes BG GB KN NK NT TN ZZ are never used
#   hint: a negative lookahead (?!BG|GB|...) handles the banned prefixes
NINO_PATTERN = r"\b(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"

# UK sort code: three pairs of digits separated by hyphens, e.g. 20-45-77
SORT_CODE_PATTERN = r"\b\d{2}-\d{2}-\d{2}\b"

# UK bank account number: exactly 8 digits on their own, e.g. 41234567
ACCOUNT_NUMBER_PATTERN = r"\b\d{8}\b"

# Email (given): good enough for redaction, not for validation
EMAIL_PATTERN = r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"

PII_PATTERNS = {
    "uk_nino": NINO_PATTERN,
    "uk_sort_code": SORT_CODE_PATTERN,
    "uk_account_number": ACCOUNT_NUMBER_PATTERN,
    "email": EMAIL_PATTERN,
}


def _find(pii_type: str, text: str) -> list[dict]:
    """All matches of one PII type, as PIIMatch dicts (the shape PIIMiddleware detectors return)."""
    pattern = re.compile(PII_PATTERNS[pii_type], re.IGNORECASE)
    return [{"type": pii_type, "value": m.group(0), "start": m.start(), "end": m.end()} for m in pattern.finditer(text)]


def find_pii(text: str) -> list[dict]:
    """Every PII match of every type, sorted by position."""
    matches = []
    for pii_type in PII_PATTERNS:
        matches.extend(_find(pii_type, text))

    matches.sort(key=lambda match: match["start"])
    return matches


def redact(text: str) -> str:
    """Replace every match from find_pii(text) with "[REDACTED_<TYPE>]", type upper-cased,
    e.g. "[REDACTED_UK_NINO]" (the same format LangChain's PIIMiddleware uses).
    Hint: replace from the END of the string backwards, so earlier start/end offsets stay valid."""
    matches = find_pii(text)
    for match in reversed(matches):
        start, end, pii_type = match["start"], match["end"], match["type"].upper()
        text = text[:start] + f"[REDACTED_{pii_type}]" + text[end:]
    return text


def redact_at_boundary(user_text: str) -> HumanMessage:
    """Use this BEFORE graph.invoke: raw PII never enters the graph, so never reaches a checkpoint."""
    return HumanMessage(redact(user_text))


def redact_pii(state) -> dict:
    """Graph node (defence in depth). For every HumanMessage in state["messages"] whose content
    contains PII, return a NEW HumanMessage with the redacted content and the SAME id.
    Same id = add_messages REPLACES the original instead of appending (Lesson 2!).
    Return {"messages": [the replacements]} (an empty list if there was nothing to redact)."""
    out = []
    for m in state["messages"]:
        if isinstance(m,HumanMessage):
            redacted_msg=redact(m.content)
            if redacted_msg != m.content:
                out.append(HumanMessage(content=redacted_msg,id=m.id)
                           )
    return {"messages":out}


# ── For create_agent (Lesson 6's investigator) ─────────────────────────────


def detector(pii_type: str):
    """A detector function in the shape PIIMiddleware expects: content -> list of PIIMatch."""
    return lambda content: _find(pii_type, content)


def pii_middleware() -> list:
    """Return one PIIMiddleware per type in PII_PATTERNS:
        PIIMiddleware(pii_type, detector=detector(pii_type), strategy="redact",
                      apply_to_input=True, apply_to_tool_results=True)
    apply_to_tool_results is the important one: the credit bureau returns bank details
    the model must never see."""
    middlewares=[]
    for pattern in PII_PATTERNS:
        middlewares.append(PIIMiddleware(pii_type=pattern,detector=detector(pattern),strategy="redact",apply_to_input=True,apply_to_tool_results=True))
    return middlewares