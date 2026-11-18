# 🧪 Testing

The unit-test suite uses **fake models only**: it's free, deterministic, runs offline, and needs no secrets, so
CI runs it on every push. Real-model checks live in two separate places: opt-in **live tests** and the
**eval suite** ([11](11_EVALS.md)).

```bash
make check                 # format, lint, then the whole suite
uv run pytest -q           # just the tests
uv run pytest tests/graph  # one layer
```

Current count: **155 passed, 11 skipped** (the skipped ones are live or Postgres tests).

---

## 🗂️ Layout: tests mirror `src/`

| Folder | Tests | Covers |
| :--- | :--- | :--- |
| `tests/api/` | `test_app.py` | Routes, status codes, the `503` handler, content-filter recovery end to end |
| `tests/agents/` | `test_risk_agent.py`, `test_prompts.py` | Tools, middleware order, retries, `risk_verdict`, prompt files |
| `tests/config/` | `test_settings.py` | Defaults, variable names, secret hiding, `get_model` errors |
| `tests/graph/` | `test_pipeline.py`, `test_intake.py`, `test_screening.py`, `test_specialists.py`, `test_underwriting.py`, `test_underwriting_hold.py`, `test_cases.py` | Every node, router and case operation |
| `tests/integrations/` | `test_locks.py`, `test_storage.py` | Locks (`409`, double-click runs once), cases surviving a restart |
| `tests/services/` | `test_policy.py`, `test_pii.py`, `test_grounding.py` | The plain-Python rules |
| `tests/` | `test_architecture.py`, `test_evals.py` | Layer rules, the evaluators |

`pyproject.toml` runs pytest with `--import-mode=importlib` (so test files in different folders can share a name
without `__init__.py` files) and `pythonpath = ["tests"]` (so any test can `import helpers`).

---

## 🎭 The fakes (`tests/conftest.py`)

| Fixture | Class | Stands in for | Use it for |
| :--- | :--- | :--- | :--- |
| `fake_structured_model` | `FakeStructuredModel(answer)` | A model's `with_structured_output()`: always returns `answer`, records what it was sent in `.sent` | One extraction |
| `scripted_structured_model` | `ScriptedStructuredModel(answers)` | Same, but one answer per call (last repeats); `.all_sent` records every call | Multi-turn cases |
| `scripted_model` | `ScriptedChatModel(script)` | A real `BaseChatModel` that replays a script of `AIMessage`s (tool calls, then an answer); records `.calls` and `.bound_tools` | Agents: `create_agent` needs a real chat-model class |

```python
def test_clean_application_is_approved(fake_structured_model):
    graph = build_graph(fake_structured_model(PRIYA), InMemorySaver())
    state = continue_application(graph, "case-1", "Priya Shah, 34, employed nurse, £42,000, want £8,000")
    assert state["outcome"] == "approved_in_principle"
```

*Gotcha:* the scripted models return a **copy with a fresh id** on every call. `add_messages` replaces messages
that share an id, so replaying the same object would overwrite history instead of appending.

Other fakes live next to the tests that need them, e.g. a model that raises `content_filter` like Azure
(`tests/api/test_app.py`), a slow model for lock races (`tests/integrations/test_locks.py`), and a
`lookup_credit_file` that always raises (`tests/agents/test_risk_agent.py`).

### `tests/helpers.py`
Read-only helpers for inspecting a case: `current_outcome(graph, case_id)` and `decisions_made(graph, case_id)`
(the outcome of every completed run, oldest first).

---

## 🔴 Live tests (real model)

Tests marked `@pytest.mark.live` call Azure OpenAI. `conftest.py` skips them unless `LOANLENS_LIVE=1`:

```bash
LOANLENS_LIVE=1 uv run pytest tests/graph/test_intake.py -m live
LOANLENS_LIVE=1 uv run pytest tests/agents/test_risk_agent.py -m live
```

They check behaviour fakes can't: monthly-to-annual income, not inventing an age, not inferring employment from a
job title (and extracting it when it is stated), and the risk agent handling a credit-bureau outage.

## 🐘 Postgres tests

Skipped unless `LOANLENS_PG_URL` is set:

```bash
LOANLENS_PG_URL=postgresql://postgres:dev@localhost:5432/postgres uv run pytest tests/integrations -v
```

---

## 🏛️ Architecture tests

`tests/test_architecture.py` parses imports with `ast` and fails if `services/` or `domain/` import a framework,
or an inner layer imports an outer one. See [02](02_PROJECT_STRUCTURE.md).

---

## ✍️ Writing a new test

1. Put it in the folder that mirrors the module under test.
2. Build the **real** graph with fakes (`build_graph(fake, InMemorySaver())`) rather than mocking nodes: test
   behaviour, not wiring.
3. Pure rules (`services/`) need no graph at all; call the function.
4. For anything that calls the real model, mark it `@pytest.mark.live`, or make it an eval.
5. Give every tool a test for success, for a bad input and for an upstream failure.

---

## 🤖 CI

`.github/workflows/publish-image.yml` runs on every push to `main`: `uv sync --frozen`, ruff lint and format
check, then `uv run pytest -q`. The Docker image is built and published **only if the tests pass**.
