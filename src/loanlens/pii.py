"""PII guardrails. Deterministic code, never an LLM.

UK GDPR data minimisation: the model gets only the personal data it needs (name, age, income, loan
amount, employment), never NI numbers, sort codes, account numbers or email addresses.

    redact_at_boundary(text)   before graph.invoke, so raw PII never reaches a checkpoint
    redact_pii (graph node)    first node of the pipeline (defence in depth)
    pii_middleware()           PIIMiddleware for agents, which also scrubs tool results
"""

import re
from collections.abc import Callable

from langchain.agents.middleware import PIIMiddleware
from langchain_core.messages import HumanMessage

from loanlens.state import PipelineState

# National Insurance number (HMRC NIM39110): 2 letters, 6 digits, a suffix letter A-D.
# People often type spaces: "AB 12 34 56 C".
#   - D F I Q U V are never the 1st or 2nd letter; O is never the 2nd letter
#   - prefixes BG GB KN NK NT TN ZZ are never used
NINO_PATTERN = (
    r"\b(?!BG|GB|KN|NK|NT|TN|ZZ)[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"
)

# UK sort code: three pairs of digits separated by hyphens, e.g. 20-45-77
SORT_CODE_PATTERN = r"\b\d{2}-\d{2}-\d{2}\b"

# UK bank account number: exactly 8 digits on their own, e.g. 41234567
ACCOUNT_NUMBER_PATTERN = r"\b\d{8}\b"

# Email: good enough for redaction, not for validation
EMAIL_PATTERN = r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"

# PII type -> compiled pattern. The type names are the ones PIIMiddleware uses in its placeholders.
PII_PATTERNS = {
    "uk_nino": re.compile(NINO_PATTERN, re.IGNORECASE),
    "uk_sort_code": re.compile(SORT_CODE_PATTERN),
    "uk_account_number": re.compile(ACCOUNT_NUMBER_PATTERN),
    "email": re.compile(EMAIL_PATTERN, re.IGNORECASE),
}


def find_pii_of_type(pii_type: str, text: str) -> list[dict]:
    """All matches of one PII type, as dicts in the shape PIIMiddleware detectors return."""
    pattern = PII_PATTERNS[pii_type]
    return [
        {"type": pii_type, "value": match.group(0), "start": match.start(), "end": match.end()}
        for match in pattern.finditer(text)
    ]


def find_pii(text: str) -> list[dict]:
    """Every PII match of every type, sorted by position."""
    matches = []
    for pii_type in PII_PATTERNS:
        matches.extend(find_pii_of_type(pii_type, text))
    matches.sort(key=lambda match: match["start"])
    return matches


def redact(text: str) -> str:
    """Replace every match with [REDACTED_<TYPE>], the format LangChain's PIIMiddleware uses."""
    # Replace from the end backwards, so earlier start/end positions stay valid.
    for match in reversed(find_pii(text)):
        placeholder = f"[REDACTED_{match['type'].upper()}]"
        text = text[: match["start"]] + placeholder + text[match["end"] :]
    return text


def redact_at_boundary(user_text: str) -> HumanMessage:
    """Redact text before it enters the graph, so raw PII never reaches a checkpoint."""
    return HumanMessage(redact(user_text))


def redact_pii(state: PipelineState) -> dict:
    """Node: replace each human message that contains PII with a redacted copy.

    The copy keeps the same id, so add_messages replaces the original instead of appending.
    """
    redacted_messages = []
    for message in state["messages"]:
        if not isinstance(message, HumanMessage):
            continue
        redacted_text = redact(message.content)
        if redacted_text != message.content:
            redacted_messages.append(HumanMessage(content=redacted_text, id=message.id))
    return {"messages": redacted_messages}


# ── For create_agent agents ────────────────────────────────────────────────


def make_detector(pii_type: str) -> Callable[[str], list[dict]]:
    """A detector function in the shape PIIMiddleware expects: text -> list of matches."""

    def detect(text: str) -> list[dict]:
        return find_pii_of_type(pii_type, text)

    return detect


def pii_middleware() -> list[PIIMiddleware]:
    """One PIIMiddleware per PII type, applied to model input and to tool results.

    Tool results matter: the credit bureau returns bank details the model must never see.
    """
    return [
        PIIMiddleware(
            pii_type=pii_type,
            detector=make_detector(pii_type),
            strategy="redact",
            apply_to_input=True,
            apply_to_tool_results=True,
        )
        for pii_type in PII_PATTERNS
    ]
