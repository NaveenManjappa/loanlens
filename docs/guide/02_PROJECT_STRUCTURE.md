# 🗂️ Project Structure: Where Code Goes

LoanLens is organised in **layers**. Each folder under `src/loanlens/` has one job, and dependencies only point
inward. This makes each piece testable on its own and makes it obvious where new code belongs.

---

## 🧅 The layers

```mermaid
graph TD
    main[main.py<br/>startup + wiring] --> api
    api[api/<br/>routes, schemas] --> graph
    graph[graph/<br/>state, nodes, routing, cases] --> agents
    graph --> services
    graph --> integrations
    agents[agents/<br/>one module per agent] --> tools
    agents --> prompts[(prompts/*.md)]
    tools[tools/<br/>thin @tool wrappers] --> services
    tools --> integrations
    services[services/<br/>plain-Python rules] --> domain
    integrations[integrations/<br/>model, storage, outside systems] --> config
    graph --> domain[domain/<br/>Pydantic models]
    config[config/<br/>settings]
```

| Folder | Holds | Must not hold |
| :--- | :--- | :--- |
| `api/` | Routes (`app.py`), request/response models (`schemas.py`) | Business logic |
| `graph/` | Pipeline state, nodes, routing, and operations on a case | SQL, HTTP clients, policy thresholds |
| `agents/` | One module per agent: its prompt, tools, middleware and how its answer is read | Shared business logic |
| `tools/` | Thin `@tool` wrappers the model can call | Real logic (that lives in `services/` and `integrations/`) |
| `services/` | Policy rules and calculations | LangChain, LangGraph, FastAPI or OpenAI imports |
| `integrations/` | Clients for outside systems: model provider, database, locks, bureau, registries | Business rules |
| `domain/` | Pydantic models shared across the app | Framework imports |
| `config/` | Settings | Secrets in code |
| `prompts/` | Prompt text (`.md`) | Code |

---

## 📄 Every module, in one line

| Module | What it does |
| :--- | :--- |
| `main.py` | Loads settings and `.env`, configures logging, opens storage and builds the graph at startup |
| `api/app.py` | `create_app()`: routes, dependency getters, `409`/`503` error handlers, `case_view()` |
| `api/schemas.py` | `MessageIn`, `DecisionIn`, `CaseView` |
| `config/settings.py` | `Settings` (pydantic-settings) and `get_settings()` |
| `domain/application.py` | `ApplicationDetails`, `EmploymentStatus` |
| `graph/state.py` | `PipelineState` and the `merge_reports` reducer |
| `graph/pipeline.py` | `build_graph()`, `start_run`, routers, the four outcome nodes |
| `graph/intake.py` | `redact_at_boundary`, `redact_pii` node, `make_extract_node`, `ground_age` node |
| `graph/screening.py` | `validate_application`, the four `check_*` nodes, `combine` |
| `graph/specialists.py` | Runs an agent as a node with private context; `default_specialists()` |
| `graph/underwriting.py` | `underwriter_review` (interrupt), `route_after_review`, `underwriter_decline`, `review_payload` |
| `graph/cases.py` | What the API calls: `continue_application`, `hold_for_underwriter`, `submit_decision`, `awaiting_underwriter`, content-filter recovery |
| `agents/risk_agent.py` | `build_risk_agent()`, `on_tool_error`, `risk_verdict` |
| `agents/compliance_agent.py` | `build_compliance_agent()` |
| `agents/middleware.py` | `pii_middleware()`: one `PIIMiddleware` per PII type |
| `agents/prompts.py` | `load_prompt(name)` via `importlib.resources` |
| `tools/risk_tools.py` | `lookup_credit_file`, `verify_self_employed_income`, `calculate_affordability` |
| `tools/compliance_tools.py` | `check_sanctions_list`, `check_vulnerability_markers` |
| `services/policy.py` | Every lending threshold and rule; `debt_to_income()` |
| `services/pii.py` | PII regexes, `find_pii`, `redact` |
| `services/grounding.py` | `age_in_text`, shared by the guard and the evaluator |
| `services/decisions.py` | `validate_decision` |
| `integrations/llm.py` | `get_model()`, `v1_base_url()`, `is_content_filter_error()` |
| `integrations/storage.py` | `open_backend()`: checkpointer + locks for SQLite or Postgres |
| `integrations/locks.py` | `CaseBusy`, `InProcessCaseLocks`, `PostgresCaseLocks` |
| `integrations/credit_bureau.py` | Fake credit bureau and HMRC income data |
| `integrations/compliance_registry.py` | Fake sanctions list and vulnerability markers |
| `evals.py` | Eval runner (`run_case`) and the four evaluators |

---

## 🛡️ The rules are enforced, not just written down

`tests/test_architecture.py` parses every module's imports and fails the build if:

- anything in `services/` or `domain/` imports `langchain`, `langchain_core`, `langchain_openai`, `langgraph`,
  `fastapi` or `openai`;
- `services/`, `integrations/` or `tools/` import an outer layer (`api`, `graph`, `agents`);
- `domain/` imports anything from `loanlens`.

*Why?* Layer rules decay quietly. A test makes the first violation visible in the pull request that adds it.

---

## 🧭 Recipes: where does my change go?

| I want to… | Change |
| :--- | :--- |
| Change a lending threshold (e.g. max loan-to-income) | `services/policy.py`, and its test in `tests/services/test_policy.py` |
| Add a new policy check | A pure function in `services/policy.py` → a node in `graph/screening.py` → add it to `SCREENING_CHECKS` and `build_graph()` in `graph/pipeline.py` |
| Add a field the applicant must give | `domain/application.py` + `REQUIRED_FIELDS` in `services/policy.py` + the extraction prompt |
| Edit what the model is told | `prompts/*.md` (no code change needed) |
| Give an agent a new tool | Data access in `integrations/`, maths in `services/`, a thin `@tool` in `tools/`, add it to that agent's tool list |
| Add a new specialist agent | `agents/<name>_agent.py` + prompt + tools, then register it in `default_specialists()` in `graph/specialists.py` |
| Add an endpoint | Schema in `api/schemas.py`, route in `api/app.py` calling a function in `graph/cases.py` |
| Read a new environment variable | A field on `Settings` in `config/settings.py`, documented in [09](09_ENVIRONMENT_VARIABLES.md) and `.env.example` |
| Swap the model provider | `integrations/llm.py` only |

---

## 🧪 Tests mirror the source

```
tests/
  conftest.py            fake models + the live-test switch
  helpers.py             read-only case inspectors (current_outcome, decisions_made)
  test_architecture.py   layer rules
  test_evals.py          evaluators
  api/ agents/ config/ graph/ integrations/ services/   one test file per module
```

See [10_TESTING.md](10_TESTING.md).
