"""Evals for LoanLens: a labelled dataset, a runner and evaluators.

run_case runs one case through the real pipeline and records the final state, every prompt sent to
any model (including inside agents), token usage and the number of model calls. Evaluators score a
run as True (pass), False (fail) or None (not applicable), and summarise turns repeated runs into
pass rates. Run: uv run python scripts/run_evals.py
"""

import json
import uuid
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

from loanlens.grounding import age_in_text
from loanlens.pii import redact_at_boundary

DATASET = Path(__file__).resolve().parents[2] / "evals" / "dataset.json"


def load_dataset() -> list[dict]:
    return json.loads(DATASET.read_text())


# ── Recording what happened (observability without any external service) ──


class PromptRecorder(BaseCallbackHandler):
    """A LangChain callback: sees every chat-model call anywhere in the graph, incl. inside agents."""

    def __init__(self):
        self.prompts: list[str] = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.prompts.append("\n".join(str(m.content) for batch in messages for m in batch))


def run_case(case: dict, build_graph) -> dict:
    """Run one case end to end on build_graph(checkpointer=...) and record what happened."""
    graph = build_graph(checkpointer=InMemorySaver())
    prompts, usage = PromptRecorder(), UsageMetadataCallbackHandler()
    thread = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    # Callbacks passed at invoke time flow down to every model call in the run, agents included.
    # Same boundary redaction as continue_application.
    final = graph.invoke(
        {"messages": [redact_at_boundary(case["text"])]},
        {
            "configurable": {"thread_id": thread},
            "recursion_limit": 100,
            "callbacks": [prompts, usage],
            # Tags and metadata show up on the trace in LangSmith (if tracing is on), so you can filter
            "tags": ["eval"],
            "metadata": {"eval_case": case["id"]},
        },
    )
    total_tokens = sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values())
    return {
        "final": final,
        "outcome": final.get("outcome"),
        "application": final.get("application") or {},
        "prompts": prompts.prompts,
        "model_calls": len(prompts.prompts),
        "total_tokens": total_tokens,
    }


# ── Evaluators: (case, run) -> True (pass) / False (fail) / None (not applicable) ──


def outcome_matches(case: dict, run: dict) -> bool | None:
    """Pass when the outcome equals expect_outcome, or differs from expect_not_outcome."""
    if "expect_outcome" in case:
        return run["outcome"] == case["expect_outcome"]
    if "expect_not_outcome" in case:
        return run["outcome"] != case["expect_not_outcome"]
    return None


def no_raw_pii_sent_to_models(case: dict, run: dict) -> bool | None:
    """Pass when none of the case's pii_strings appears in any prompt sent to any model."""
    if "pii_strings" not in case:
        return None
    for prompt in run["prompts"]:
        for pii in case["pii_strings"]:
            if pii in prompt:
                return False
    return True


def age_not_invented(case: dict, run: dict) -> bool | None:
    """Reference-free grounding check: an extracted age must appear in the applicant's text."""
    age = run["application"].get("age")
    text = case["text"]
    return age_in_text(age, text)


def extraction_correct(case: dict, run: dict) -> bool | None:
    """Reference-based: each expect_<field> (other than the outcome keys) matches the extracted field."""
    expect_keys = [k for k in case if k.startswith("expect_") and k not in ("expect_outcome", "expect_not_outcome")]
    if not expect_keys:
        return None
    for expect_key in expect_keys:
        field = expect_key[len("expect_"):]
        expected_value = case[expect_key]
        actual_value = run["application"].get(field)
        if isinstance(expected_value, (int, float)) and isinstance(actual_value, (int, float)):
            if float(expected_value) != float(actual_value):
                return False
        else:
            if expected_value != actual_value:
                return False
    return True


EVALUATORS = [outcome_matches, no_raw_pii_sent_to_models, age_not_invented, extraction_correct]


def summarise(results: list[dict]) -> dict:
    """Pass rate per evaluator, passes / (passes + fails), ignoring not-applicable runs, to 2 dp."""
    summary = {}
    for evaluator in EVALUATORS:
        name = evaluator.__name__
        applicable_runs = [r for r in results if r["scores"].get(name) is not None]
        if not applicable_runs:
            continue
        passes = sum(1 for r in applicable_runs if r["scores"][name] is True)
        fails = sum(1 for r in applicable_runs if r["scores"][name] is False)
        pass_rate = passes / (passes + fails) if (passes + fails) > 0 else 0.0
        summary[name] = round(pass_rate, 2)
    return summary

