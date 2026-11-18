"""Prompts are files in loanlens/prompts/, read through load_prompt."""

from loanlens.agents.compliance_agent import COMPLIANCE_PROMPT
from loanlens.agents.prompts import load_prompt
from loanlens.agents.risk_agent import REFER, SYSTEM_PROMPT


def test_every_prompt_file_loads_and_is_not_empty():
    for name in ("extraction.md", "risk_agent.md", "compliance_agent.md"):
        assert load_prompt(name).strip()


def test_risk_prompt_names_the_verdict_the_code_falls_back_to():
    # risk_verdict returns REFER when the credit check fails; the model must know the same word.
    assert REFER in SYSTEM_PROMPT


def test_compliance_prompt_lists_its_three_verdicts():
    for verdict in ("COMPLIANT", "NEEDS REVIEW", "NOT COMPLIANT"):
        assert verdict in COMPLIANCE_PROMPT
