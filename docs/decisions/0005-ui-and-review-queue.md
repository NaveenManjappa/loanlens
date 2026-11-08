# ADR 0005: A two-screen UI and a review queue

**Status:** Proposed. Not built. One open question: the UI framework (see Decision).
**Date:** November 2026

## Context

LoanLens is API-only. Showing it, in a demo video or an interview, means curl commands. Two kinds of
user need a screen:

- **Applicants:** send a message and see LoanLens's reply.
- **Underwriters:** see the cases waiting for them, read the specialists' reports, and decide.

The second one has a gap underneath it: **there's no way to list cases.** The API can read one case by
id, but nothing answers "which cases are waiting for an underwriter?". Cases live in the LangGraph
checkpointer, which stores state by thread and can't be queried by status. Scanning every thread's
latest checkpoint would work for a demo, but it gets slower with every case, and it couples the UI to
LangGraph's storage format.

## Options considered

**For the review queue:**
1. **A small `cases` table (chosen).** One row per case: `case_id`, `status` (the outcome),
   `awaiting_underwriter`, `created_at`, `updated_at`. The API updates it after each run and decision.
   It's indexed and cheap to query, and it's our own schema.
2. **Scan checkpoints.** Rejected: cost grows with every case, and it depends on internal storage details.
3. **LangGraph's Store** (`BaseStore`, for example `PostgresStore`). A key-value store with search. It would
   work, but a plain table with an index is simpler for a status list, and familiar to anyone reading the code.

**For the UI:**
1. **Streamlit.** Python, fastest to build, good enough for a demo. Limited control over layout and behaviour.
2. **Angular single-page app.** Shows end-to-end ownership (the API plus a real front end) and builds on
   existing Angular experience. More setup: routing, an HTTP client, the sign-in flow.
3. **Server-rendered pages from FastAPI (Jinja + HTMX).** One deployable and little JavaScript, but a
   third style to learn and maintain.

## Decision

- **Add the `cases` table** (SQLite locally, Postgres deployed), written by the API after every run and
  decision, and an underwriter-only `GET /reviews` endpoint listing cases awaiting review, oldest first.
- **Two screens, nothing more:**

  | Screen | Who | What it does |
  |---|---|---|
  | Application chat | applicant | Start a case (`POST /cases`), send messages, show replies and the current status |
  | Review queue | underwriter (signed in) | List pending cases; open one to see the application, risk flags, specialist reports and any applicant messages held during review; approve or decline with a reason |

- **Rules for the UI:**
  - it talks only to the API, never to the database;
  - it only shows what the API returns, which is already redacted;
  - case ids never appear next to personal data in URLs or logs;
  - every changing action sends an `Idempotency-Key` (ADR 0003);
  - the review queue uses the Entra ID sign-in (ADR 0004).
- **Open question: the framework.** Choose Streamlit if the main goal is the demo video. Choose Angular if
  the goal is to show full-stack ownership for Forward Deployed Engineer roles. Decide before starting,
  and record the choice here.

## Consequences

- The underwriter workflow becomes demonstrable end to end, and the demo video becomes possible.
- The `cases` table is a second source of truth beside the checkpoints. It must be updated in the same
  place as every outcome change (in `cases.py` and the decision route), or the two will drift apart.
  The checkpoint stays the authority; the table is an index.
- Depends on ADR 0003 (keys) and ADR 0004 (sign-in): build those first.
- One more component to deploy: a second container app, or static hosting for an Angular build.

## Revisit when

- underwriters need search, filters or assignment (who works which case);
- the `cases` table starts collecting more than an index needs: that's the start of the audit log from
  ADR 0002, and it should be designed as one.
