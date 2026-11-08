"""The risk specialist agent and its tools.

The model never runs a tool itself: it asks for one, the agent runs it and returns a ToolMessage.
Each tool's docstring is the description the model reads. The tools fake external systems (a
credit bureau, HMRC income data) with canned data.

Middleware supplies the guardrails: PII redaction (also of tool results), retries for transient tool
failures, a readable error for the model when retries run out, and a cap on model calls.
risk_verdict then applies policy in code to the agent's finished conversation.
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.state import CompiledStateGraph

from loanlens.pii import pii_middleware

MAX_MODEL_CALLS = 6  # up to 5 tool rounds + a final answer
MAX_DEBT_TO_INCOME = 0.4  # monthly debts incl. the new loan <= 40% of gross monthly income
TRANSIENT_ERRORS = (ConnectionError, TimeoutError)  # worth retrying: the network, not our code
REFER = "REFER TO SENIOR UNDERWRITER"

SYSTEM_PROMPT = f"""You are a loan risk investigator at a UK lender.
Investigate the referred application with your tools, then reply with a short recommendation.
The FIRST line must be exactly one of: APPROVE, DECLINE, {REFER}.
Then 2-4 bullet points of evidence. Only use facts returned by your tools or present in the
application. If a tool reports it is unavailable, say so; never guess the missing data.
Call independent tools together in one step."""

# ── Tools (fake external systems: in real life a credit bureau API, HMRC/bank data) ──

_CREDIT_FILES = {
    "priya shah": {"credit_score": 870, "missed_payments_12m": 0, "existing_monthly_debt": 150},
    "sam jones": {
        "credit_score": 640,
        "missed_payments_12m": 2,
        "existing_monthly_debt": 450,
        # Real bureau files carry bank details. The model must never see these.
        "bank_details": {"sort_code": "20-45-77", "account_number": "41234567"},
    },
    "tom reid": {"credit_score": 910, "missed_payments_12m": 0, "existing_monthly_debt": 0},
}
_VERIFIED_SELF_EMPLOYED_INCOME = {
    "sam jones": {"tax_year_2024_25": 31_000, "tax_year_2025_26": 36_500},
    "tom reid": {"tax_year_2024_25": 58_000, "tax_year_2025_26": 61_000},
}


@tool
def lookup_credit_file(applicant_name: str) -> dict:
    """Look up the applicant's credit file at the credit bureau.

    Returns credit_score (0-999, higher is better), missed_payments_12m, and
    existing_monthly_debt in GBP. Use this for every investigation."""
    record = _CREDIT_FILES.get(applicant_name.strip().lower())
    return record or {"error": f"No credit file found for '{applicant_name}'"}


@tool
def verify_self_employed_income(applicant_name: str) -> dict:
    """Fetch verified annual income (GBP) from the last two tax years for a
    SELF-EMPLOYED applicant. Only useful when employment_status is self_employed."""
    record = _VERIFIED_SELF_EMPLOYED_INCOME.get(applicant_name.strip().lower())
    return record or {"error": f"No verified self-employed income for '{applicant_name}'"}


@tool
def calculate_affordability(
    annual_income: float, existing_monthly_debt: float, loan_amount: float, term_months: int = 36
) -> dict:
    """Check whether the new loan is affordable under LoanLens policy.

    Args:
        annual_income: gross annual income in GBP
        existing_monthly_debt: current monthly debt repayments in GBP
        loan_amount: total amount to borrow in GBP
        term_months: repayment term in months (default 36)

    Returns monthly_repayment, debt_to_income, and affordable (bool)."""
    if annual_income <= 0 or term_months <= 0:
        return {"error": "annual_income and term_months must be positive"}
    monthly_repayment = round(loan_amount / term_months, 2)  # no interest in policy v0
    monthly_income = annual_income / 12
    debt_to_income = round((existing_monthly_debt + monthly_repayment) / monthly_income, 2)
    return {
        "monthly_repayment": monthly_repayment,
        "debt_to_income": debt_to_income,
        "affordable": debt_to_income <= MAX_DEBT_TO_INCOME,
    }


TOOLS = [lookup_credit_file, verify_self_employed_income, calculate_affordability]

# ── The agent ──────────────────────────────────────────────────────────────


def on_tool_error(exc: Exception, request) -> str | None:
    """Turn a transient tool failure into a message for the model; return None to let anything else raise.

    The message names the exception type, never its text, which could leak internal details.
    """
    if isinstance(exc, TRANSIENT_ERRORS):
        tool_name = request.tool_call["name"]
        return f"`{tool_name}` is unavailable right now ({type(exc).__name__}). Do not guess its data."
    return None


def build_risk_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    retry_initial_delay: float = 1.0,
) -> CompiledStateGraph:
    """Build the risk agent (tools default to TOOLS).

    Middleware order matters: retries sit inside error handling, so only the final failure reaches
    on_tool_error.
    """
    return create_agent(
        model=model,
        tools=tools if tools is not None else TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            *pii_middleware(),
            ToolErrorMiddleware(on_error=on_tool_error),
            ToolRetryMiddleware(
                max_retries=2,
                retry_on=TRANSIENT_ERRORS,
                on_failure="error",
                initial_delay=retry_initial_delay,
            ),
            ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end"),
        ],
    )


def risk_verdict(messages: list[BaseMessage]) -> str:
    """The risk agent's final recommendation, with policy applied in code.

    Never recommend approval without a credit check: if lookup_credit_file failed, refer.
    Otherwise use the first line of the agent's answer.
    """
    for message in messages:
        credit_check_failed = (
            isinstance(message, ToolMessage)
            and message.name == "lookup_credit_file"
            and message.status == "error"
        )
        if credit_check_failed:
            return REFER

    content = str(messages[-1].content).strip()
    if not content:
        return REFER
    return content.splitlines()[0].strip()
