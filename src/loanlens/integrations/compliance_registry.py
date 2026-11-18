"""A fake sanctions list (HM Treasury / OFSI) and CRM vulnerability markers, with canned data.

Sanctions screening of every applicant is plain code (graph.screening.check_sanctions calls
is_sanctioned); the compliance agent's tool repeats the check for its report.
"""

_SANCTIONS_LIST = {"ivan petrov"}
_VULNERABILITY_MARKERS = {"tom reid": ["recent bereavement (self-declared)"]}


def _normalise(name: str) -> str:
    return name.strip().lower()


def is_sanctioned(applicant_name: str | None) -> bool:
    """True if the name is on the sanctions list.

    Name-only matching can produce false positives, so a match must go to a human, never straight
    to approve or decline.
    """
    if not applicant_name:
        return False
    return _normalise(applicant_name) in _SANCTIONS_LIST


def vulnerability_markers(applicant_name: str) -> list[str]:
    """Recorded vulnerability markers that need extra care under Consumer Duty."""
    return _VULNERABILITY_MARKERS.get(_normalise(applicant_name), [])
