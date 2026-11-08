# ADR 0003: Idempotency keys for requests that change a case

**Status:** Proposed. Not built.
**Date:** November 2026

## Context

Two endpoints change a case: `POST /cases/{case_id}/messages` (an applicant message) and
`POST /cases/{case_id}/decision` (an underwriter decision). Each one runs the graph, calls the model
(1 to 6 calls), and writes checkpoints.

The per-case lock (`locks.py`) stops two requests changing the same case **at the same time**: the
second gets `409`. It does nothing for the same request arriving **again after the first finished**.
That happens whenever a client retries:

- a network timeout after the server has already processed the message, so the client retries;
- a user who refreshes and resubmits;
- a mobile app that resends when it reconnects.

Each retry runs the graph again. The case history gets a second copy of the message, the model is paid
for twice, and if a later version of LoanLens triggers external actions (an offer email, a payout), they
happen twice too.

## Options considered

1. **Idempotency-Key header (chosen).** The client generates a random key per user action and sends it
   with the request. The server stores the key with its response, and a repeat of the key returns the
   stored response without running anything. This is the pattern in the IETF HTTP API working group's
   [Idempotency-Key header draft](https://www.ietf.org/archive/id/draft-ietf-httpapi-idempotency-key-header-07.html),
   and the one payment APIs use.
2. **Deduplicate by message content.** Rejected: an applicant can legitimately send the same words twice
   ("yes", "34"), and content alone can't tell a retry from a new message.
3. **LangGraph Agent Server's multitask strategies** (reject, enqueue, interrupt, rollback). Not
   available: LoanLens runs its own FastAPI app, not the hosted server. Those strategies also handle
   concurrent runs, which the lock already covers, rather than retries.

## Decision

- Both changing endpoints accept an `Idempotency-Key` header: a UUID that the client generates once per
  user action and reuses for every retry of that action.
- A key is scoped to its case and endpoint: the same key on another case is a different key.
- A table `idempotency_keys`: `case_id`, `endpoint`, `key`, `request_hash` (a hash of the body),
  `status_code`, `response_body`, `created_at`, with primary key (`case_id`, `endpoint`, `key`).
  SQLite locally and Postgres when deployed, the same choice `checkpointing.py` already makes.
- Behaviour, all inside the existing per-case lock:

  | Situation | Response |
  |---|---|
  | New key | Run as normal, store the response, return it |
  | Known key, same body | Return the stored response; run nothing |
  | Known key, different body | `422`: a key can't be reused for a different request |
  | Same key while the first is still running | `409` (the lock already does this) |
  | No key | Run as today. Make the header required once the UI sends it (`400` without it) |

- Only completed responses (2xx and 4xx) are stored. A 5xx is not stored, so a retry can try again.
- Keys expire after 24 hours, removed by a clean-up at startup or on a schedule.

## Consequences

- Retries become harmless: one user action means one graph run and one charge, however often it's resent.
- **At-least-once, not exactly-once.** If the process crashes after the graph ran but before the response
  was stored, a retry runs again. That's acceptable for messages. Any future step with real-world effects
  (an email, a payout) must also be idempotent itself, for example a unique constraint on "one offer per case".
- One more table to create and clean up, and one more lookup per request (cheap next to a model call).
- The client must generate and keep a key per action. The UI (ADR 0005) does this, and API users must be told.

## Revisit when

- external side effects are added (then make them idempotent too, and consider an outbox table);
- LoanLens moves to LangGraph Agent Server, whose own run handling may replace part of this.
