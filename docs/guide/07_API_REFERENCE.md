# 🌐 API Reference

The HTTP API is a thin FastAPI layer (`api/app.py`). Routes parse input, call a function in `graph/cases.py`
and return a view. They hold no business logic. Interactive docs are served at `/docs`.

---

## 📍 Endpoints

| Method | Path | Purpose | Success | Errors |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/cases/{case_id}/messages` | An applicant sends a message | `200` `CaseView` | `409` busy · `422` · `503` |
| `GET` | `/cases/{case_id}` | Read a case (runs nothing) | `200` `CaseView` | `404` unknown case |
| `GET` | `/cases/{case_id}/review` | What the underwriter sees | `200` review payload | `404` no review pending · `409` busy |
| `POST` | `/cases/{case_id}/decision` | The underwriter decides | `200` `CaseView` | `409` busy or no review pending · `422` · `503` |
| `GET` | `/health` | Health probe | `200` `{"status": "ok"}` | – |

`case_id` must match `^[A-Za-z0-9_-]{1,64}$` (otherwise `422`). It becomes the LangGraph `thread_id`. A new id
starts a new case; there is no separate "create" call.

---

## 📦 Models (`api/schemas.py`)

### `MessageIn`
| Field | Type | Rules |
| :--- | :--- | :--- |
| `text` | string | 1–4000 characters |

### `DecisionIn`
| Field | Type | Rules |
| :--- | :--- | :--- |
| `decision` | `"approve"` \| `"decline"` | required |
| `reason` | string | non-empty |
| `underwriter_id` | string | non-empty |

### `CaseView`
What callers see of a case: never the raw state (least exposure).

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `case_id` | string | |
| `outcome` | string \| null | See the outcomes in [03](03_THE_PIPELINE.md) |
| `reply` | string \| null | LoanLens's latest message to the applicant |
| `awaiting_underwriter` | bool | `true` while the case is paused for review |

The review payload is documented in [05](05_HUMAN_IN_THE_LOOP.md).

---

## 🧪 A full session with curl

```bash
URL=http://127.0.0.1:8000

# 1. A complete, clean application → approved in principle
curl -X POST $URL/cases/demo-priya/messages -H 'content-type: application/json' \
  -d '{"text": "Hi, I am Priya Shah, 34, employed as a nurse, earning £42,000 a year. I want £8,000."}'
# {"case_id":"demo-priya","outcome":"approved_in_principle","reply":"Good news: ...","awaiting_underwriter":false}

# 2. Missing details → LoanLens asks, then the next message completes the case
curl -X POST $URL/cases/demo-tom/messages -H 'content-type: application/json' \
  -d '{"text": "Tom Reid, employed plumber, about £60k a year. Looking for £8,000."}'
# "reply": "Thanks. To continue, we need: age."
curl -X POST $URL/cases/demo-tom/messages -H 'content-type: application/json' -d '{"text": "I am 41."}'

# 3. A referral → underwriter review
curl -X POST $URL/cases/demo-sam/messages -H 'content-type: application/json' \
  -d '{"text": "Sam Jones, 29, self-employed web designer, about £40k a year, want £5,000"}'
# "outcome":"referred", "awaiting_underwriter":true
curl $URL/cases/demo-sam/review
curl -X POST $URL/cases/demo-sam/decision -H 'content-type: application/json' \
  -d '{"decision": "approve", "reason": "Two years of accounts seen", "underwriter_id": "uw-1"}'
# "outcome":"approved_in_principle", "awaiting_underwriter":false
```

---

## 🚦 Status codes and why

| Code | When | Body |
| :--- | :--- | :--- |
| `409` | Another request is changing this case right now (`CaseBusy`) | "This case is already being processed. Please wait a moment and try again." |
| `409` | `POST /decision` on a case with no pending review | "No review pending" |
| `404` | `GET /cases/{id}` for an unknown case, or `GET /review` with nothing pending | |
| `422` | Body or `case_id` fails validation, checked **before** any graph runs | FastAPI's validation detail |
| `503` | The model call failed: down, rate-limited, unreachable or misconfigured (`openai.APIError`) | "LoanLens is temporarily unavailable. Please try again shortly." The provider's error text is never returned or logged |

A message blocked by the content filter is **not** an error: it returns `200` with a calm reply (see [06](06_PII_AND_SAFETY.md)).

---

## 🔐 Concurrency inside the routes

Every route that changes a case, plus `GET /review`, runs inside `locks.hold(case_id)`:

```python
with locks.hold(case_id):
    if awaiting_underwriter(graph, case_id) is None:
        raise HTTPException(409, "No review pending")
    submit_decision(graph, case_id, body.model_dump())
```

The pending-review check is **inside** the lock: checking first and acting later would be a race. See [08](08_PERSISTENCE_AND_CONCURRENCY.md).

---

## 🧩 How the app is assembled

```python
create_app(graph=None, lifespan=None) -> FastAPI
```

- **Tests** pass a graph built on fake models: `create_app(build_graph(fake_model, InMemorySaver()))`.
- **Production** (`main.py`) passes a `lifespan` that opens storage, builds the real graph and sets
  `app.state.graph` and `app.state.locks` at startup, then closes connections at shutdown.
- Routes get the graph and locks through the FastAPI dependencies `get_graph` and `get_locks`.

> ⚠️ **No authentication yet.** Anyone with the URL can read a case or decide as any `underwriter_id`. See
> [ADR 0004](../decisions/0004-authentication-entra-id.md).
