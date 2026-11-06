"""Lesson 11: evals for LoanLens. Measure, don't guess.

An EVAL = a dataset of cases + a way to run each one + EVALUATORS that score each run.
One passing run proves little (Lesson 3), so the runner repeats every case and we report
PASS RATES, plus cost signals (tokens, model calls) per case.

run_case (done for you) runs one case through the real pipeline and records:
  final state · outcome · every prompt sent to any model · token usage · number of model calls

YOUR JOB:  uv run pytest tests/test_lesson11_evals.py -v  -> green
  1-4. the four evaluators     5. summarise
Then:  uv run python scripts/run_evals.py   (real model; ~8 cases × 2 repeats)
"""

import json
import re  # noqa: F401  (you'll want it for age_not_invented)
import uuid
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver

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
    """Run one case end to end. build_graph(checkpointer=...) -> compiled pipeline."""
    graph = build_graph(checkpointer=InMemorySaver())
    prompts, usage = PromptRecorder(), UsageMetadataCallbackHandler()
    thread = f"eval-{case['id']}-{uuid.uuid4().hex[:6]}"
    # Callbacks passed at invoke time flow down to every model call in the run, agents included.
    # Same boundary redaction as continue_application (Lesson 7).
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
    """If the case has "expect_outcome": pass when run["outcome"] equals it.
    If it has "expect_not_outcome": pass when run["outcome"] is anything ELSE.
    Neither key -> None."""
    if "expect_outcome" in case:
        return run["outcome"] == case["expect_outcome"]
    if "expect_not_outcome" in case:
        return run["outcome"] != case["expect_not_outcome"]
    return None


def no_raw_pii_sent_to_models(case: dict, run: dict) -> bool | None:
    """If the case has "pii_strings": pass when NONE of them appears in ANY of run["prompts"].
    No "pii_strings" -> None."""
    if "pii_strings" not in case:
        return None
    for prompt in run["prompts"]:
        for pii in case["pii_strings"]:
            if pii in prompt:
                return False
    return True


def age_not_invented(case: dict, run: dict) -> bool | None:
    """Grounding check (Lesson 3's 'shape, not truth'). Uses only the INPUT TEXT, no expected answer:
      extracted age is None                         -> True  (nothing invented)
      extracted age's digits appear in case["text"] -> True  (grounded, e.g. r"\\b34\\b")
      otherwise                                     -> False (the model made it up)
    Use run["application"].get("age")."""
    extracted_age = run["application"].get("age")
    if extracted_age is None:
        return True
    return re.search(rf"\b{extracted_age}\b", case["text"]) is not None


def extraction_correct(case: dict, run: dict) -> bool | None:
    """Reference-based: for every key starting with "expect_" EXCEPT expect_outcome and
    expect_not_outcome (i.e. expect_age, expect_annual_income), compare with
    run["application"][field] where field = the key without "expect_".
    Numbers: compare as floats (42000 == 42000.0). No such keys -> None. All match -> True."""
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
    """results: one dict per run: {"case_id": ..., "scores": {evaluator_name: True/False/None},
                                   "total_tokens": int, "model_calls": int}
    Return {evaluator_name: pass_rate} where pass_rate = passes / (passes + fails),
    IGNORING None (not applicable). Round to 2 dp. Skip evaluators with no applicable runs."""
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

