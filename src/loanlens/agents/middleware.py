"""Middleware shared by the agents."""

from collections.abc import Callable

from langchain.agents.middleware import PIIMiddleware

from loanlens.services.pii import PII_PATTERNS, find_pii_of_type


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
