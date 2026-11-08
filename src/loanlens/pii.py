"""PII guardrails. Deterministic code, never an LLM.

UK GDPR data minimisation: the model gets only the personal data it needs (name, age, income, loan
amount, employment), never NI numbers, sort codes, account numbers or email addresses.

    redact_at_boundary(text)   before graph.invoke, so raw PII never reaches a checkpoint
    redact_pii (graph node)    first node of the pipeline (defence in depth)
    pii_middleware()           PIIMiddleware for agents, which also scrubs tool results
"""

import re

from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage

# ── Patterns ───────────────────────────────────────────────────────────────
# All are compiled with re.IGNORECASE by _find() below.

# National Insurance number (HMRC NIM39110):
#   2 letters, 6 digits, 1 suffix letter A, B, C or D.  People often type spaces: "AB 12 34 56 C".
#   - D F I Q U V are never the 1st or 2nd letter;  O is never the 2nd letter
#   - prefixes BG GB KN NK NT TN ZZ are never used
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
    """Replace every match with [REDACTED_<TYPE>], the format LangChain's PIIMiddleware uses."""
    matches = find_pii(text)
    for match in reversed(matches):
        start, end, pii_type = match["start"], match["end"], match["type"].upper()
        text = text[:start] + f"[REDACTED_{pii_type}]" + text[end:]
    return text


def redact_at_boundary(user_text: str) -> HumanMessage:
    """Redact text before it enters the graph, so raw PII never reaches a checkpoint."""
    return HumanMessage(redact(user_text))


def redact_pii(state) -> dict:
    """Node: replace each human message that contains PII with a redacted copy.

    The copy keeps the same id, so add_messages replaces the original instead of appending.
    """
    out = []
    for m in state["messages"]:
        if isinstance(m,HumanMessage):
            redacted_msg=redact(m.content)
            if redacted_msg != m.content:
                out.append(HumanMessage(content=redacted_msg,id=m.id)
                           )
    return {"messages":out}


# ── For create_agent agents ────────────────────────────────────────────────


def detector(pii_type: str):
    """A detector function in the shape PIIMiddleware expects: content -> list of PIIMatch."""
    return lambda content: _find(pii_type, content)


def pii_middleware() -> list:
    """One PIIMiddleware per PII type, applied to model input and to tool results.

    Tool results matter: the credit bureau returns bank details the model must never see.
    """
    middlewares=[]
    for pattern in PII_PATTERNS:
        middlewares.append(PIIMiddleware(pii_type=pattern,detector=detector(pattern),strategy="redact",apply_to_input=True,apply_to_tool_results=True))
    return middlewares