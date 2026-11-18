# ⚠️ Known Gotchas

Non-obvious behaviour that has bitten, or would bite, someone changing LoanLens. Each one says what goes wrong,
why, and what the code does about it.

---

## 1. Resetting a key that has a reducer needs `Overwrite`

**The issue:** `start_run` must clear `risk_flags` at the start of each run. Returning `{"risk_flags": []}` does
nothing: the flags from the previous run survive.

**The reason:** `risk_flags` has the reducer `operator.add`. LangGraph *merges* every update through the
reducer, so `old + []` is just `old`.

**The solution:**

```python
from langgraph.types import Overwrite
return {"risk_flags": Overwrite([]), "held_messages": Overwrite([]), "specialist_reports": Overwrite({})}
```

---

## 2. Replace a message by reusing its id

**The issue:** to redact or withhold a message already in state, appending a new one leaves the original there.

**The reason:** `add_messages` appends a message with a new id, but **replaces** one whose id matches.

**The solution:** `redact_pii` and `withhold_blocked_message` both build the new `HumanMessage` with
`id=original.id`.

The flip side, in tests: the fake models return a **copy with a fresh id** on every call. Replaying the same
`AIMessage` object would silently overwrite history instead of appending.

---

## 3. A node with `interrupt()` re-runs from its first line

**The issue:** code placed before `interrupt()` in `underwriter_review` runs twice: once when the graph pauses,
and again on resume.

**The reason:** LangGraph resumes a node by re-executing it; `interrupt()` returns the resume value the second
time.

**The solution:** nothing before `interrupt()` may have side effects. `review_payload` is a pure function.

---

## 4. `update_state` hides the interrupt marker

**The issue:** after `hold_for_underwriter` adds a message to a paused case, the case no longer *looks* paused
if you only check `snapshot.tasks[...].interrupts`.

**The reason:** `update_state` writes a new checkpoint whose tasks don't carry the interrupt, but
`snapshot.next` is still `("underwriter_review",)`.

**The solution:** `awaiting_underwriter()` checks both, and rebuilds the payload from state when the marker is
gone. Any future review queue must do the same (ADR 0001).

---

## 5. A blocked message bricks a case unless it is withheld

**The issue:** after Azure's content filter blocks one message, **every** later message on that case is blocked
too, even polite ones.

**The reason:** the blocked message is already checkpointed, and extraction sends the whole conversation every
turn.

**The solution:** `withhold_blocked_message` replaces it with `[message withheld]` (same id) and records the
update as `request_missing_info`, so the next run is normal. See [06](06_PII_AND_SAFETY.md).

---

## 6. Middleware order changes behaviour

**The issue:** put `ToolErrorMiddleware` *inside* `ToolRetryMiddleware` and the first failure becomes a message
for the model: nothing is ever retried.

**The solution:** the order in `build_risk_agent` is `ToolErrorMiddleware` then `ToolRetryMiddleware`, so retries
happen inside and only the **final** failure reaches `on_tool_error`. `tests/agents/test_risk_agent.py` pins it.

---

## 7. PII can arrive from a tool, not just the user

**The issue:** redacting the applicant's text isn't enough: the credit bureau returns a sort code and account
number.

**The solution:** `PIIMiddleware(apply_to_tool_results=True)` on every agent. The
`no_raw_pii_sent_to_models` eval checks every prompt, agents included.

---

## 8. A word boundary is not enough to find an age

**The issue:** `\b34\b` matches `£34,000`, so an applicant earning £34,000 would "confirm" an invented age of 34.

**The solution:** `age_in_text` uses `(?<![£\d.,]){age}(?![.,]?\d)`: no currency sign, digit, dot or comma
before, and no digit (optionally after a dot or comma) after.

---

## 9. Never log or return a provider error's text

**The issue:** `str(openai_error)` can include parts of the prompt, i.e. customer data.

**The solution:** the `503` handler and `on_tool_error` use only `type(exc).__name__`. Keep it that way when
adding error handling.

---

## 10. On Azure, `model` is your deployment name

**The issue:** setting the model to `gpt-5` gives `DeploymentNotFound`.

**The solution:** `AZURE_OPENAI_DEPLOYMENT_NAME` is the name **you** gave the deployment in Foundry, and is passed
as `ChatOpenAI(model=...)`. The endpoint uses the v1 API (`.../openai/v1/`), so no `api_version` is needed.

---

## 11. Prompts must ship inside the package

**The issue:** reading `open("prompts/risk_agent.md")` works from the repo root and fails in Docker or from
another directory.

**The solution:** `load_prompt()` uses `importlib.resources.files("loanlens")`, and the prompts live inside
`src/loanlens/prompts/`, so the build includes them in the wheel.

---

## 12. Settings don't feed LangSmith

**The issue:** `LANGSMITH_*` values in `.env` are ignored even though `Settings` reads `.env`.

**The reason:** pydantic-settings reads `.env` into the `Settings` object only; it doesn't export to
`os.environ`, which is where the LangSmith library looks.

**The solution:** `main.py` (and `scripts/run_evals.py`) call `load_dotenv()` first.

---

## 13. Postgres behind a pooler

**The issue:** errors about prepared statements, or checkpoints that never seem to save.

**The solution:** the pool uses `prepare_threshold=0` and `autocommit=True`, and on Neon you use the **direct**
connection string (no `-pooler`) with `sslmode=require`.

---

## 14. Locks are not idempotency

A retried request **after** the first finished runs the graph again: the lock only stops overlapping requests.
See [08](08_PERSISTENCE_AND_CONCURRENCY.md) and ADR 0003.
