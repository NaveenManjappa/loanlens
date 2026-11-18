"""Intake: the first nodes of every run.

    redact_pii -> extract_details -> ground_age

redact_pii replaces PII in the applicant's messages (defence in depth: cases.py already redacts at
the boundary). extract_details is the only LLM call on the main path: the model returns structured
output (an ApplicationDetails, not prose), and every field is optional, so anything the applicant
didn't state stays None and validation asks for it instead of the model guessing. ground_age
discards an extracted age the applicant never wrote (services.grounding).
"""

from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from loanlens.agents.prompts import load_prompt
from loanlens.domain.application import ApplicationDetails
from loanlens.graph.state import PipelineState
from loanlens.services.grounding import age_in_text
from loanlens.services.pii import redact

EXTRACTION_PROMPT = load_prompt("extraction.md")


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
        redacted_text = redact(str(message.content))
        if redacted_text != message.content:
            redacted_messages.append(HumanMessage(content=redacted_text, id=message.id))
    return {"messages": redacted_messages}


def make_extract_node(model: BaseChatModel) -> Callable[[PipelineState], dict]:
    """Build the extract_details node around a model (passed in, so tests can use a fake)."""
    structured_model = model.with_structured_output(ApplicationDetails)

    def extract_details(state: PipelineState) -> dict:
        """Ask the model for ApplicationDetails and write them to state['application']."""
        details = structured_model.invoke([SystemMessage(EXTRACTION_PROMPT), *state["messages"]])
        return {"application": details.model_dump()}

    return extract_details


def applicant_text(messages: list[BaseMessage]) -> str:
    """The text of the applicant's own messages; LoanLens's replies are not evidence."""
    return "\n".join(str(m.content) for m in messages if isinstance(m, HumanMessage))


def ground_age(state: PipelineState) -> dict:
    """Node: discard the extracted age if the applicant never wrote it (otherwise an empty update)."""
    age = state["application"].get("age")
    text = applicant_text(state["messages"])
    if age_in_text(age, text):
        return {}
    return {"application": {**state["application"], "age": None}}
