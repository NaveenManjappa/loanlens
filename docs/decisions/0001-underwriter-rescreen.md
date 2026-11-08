# ADR 0001: Messages during underwriter review, and re-screening

**Status:** Part 1 implemented. Part 2 designed, not built.
**Date:** November 2026

## Context

Referred LoanLens cases pause at `underwriter_review` (`interrupt()`) until a named underwriter
submits a decision and a reason. Two gaps were found:

1. **Review bypass.** A new applicant message on a paused thread started a fresh run from `START`,
   silently abandoning the pending review. If the new run found no risk flags, the applicant was
   auto-approved **without any underwriter**, so an applicant could skip review just by sending a message.
2. **Stale facts.** The applicant may send new information (e.g. "I've got a salaried job") that
   makes the original screening out of date, including data the model can't infer (the new salary).

## Decision

### Part 1: Hold, don't run (implemented)
While a review is pending, `continue_application` does **not** invoke the graph. `hold_for_underwriter`
redacts the message (boundary redaction) and attaches it to the case with `update_state`
(`messages` + `held_messages`). The underwriter sees it as `applicant_updates` in the review payload.
Only the underwriter's decision moves the case forward.

Gotcha: `update_state` writes a new checkpoint whose `.tasks` no longer carry the interrupt marker,
while `.next` is still `('underwriter_review',)`. The review queue therefore checks `.next` too.

### Part 2: `"rescreen"` decision (designed)
- Add `"rescreen"` to the allowed underwriter decisions (still requires a reason and an underwriter id).
- `route_after_review` returns `"start_run"`, so the **same run** loops back: `start_run` clears per-run
  scratch, and re-extraction reads the whole conversation including the held updates.
- On rescreen, set `requires_underwriter = True`; `route_after_screening` then always routes to
  `refer_to_underwriter`. **A re-screened case must return to a human even if it now has no flags.**
  Otherwise this is the same bypass by a different route.
- Missing new facts (e.g. the new salary) should not be guessed: allow the underwriter to request
  specific fields, routing through `request_missing_info`.

## Consequences

- No route lets an applicant's message bypass human review (an auditable control).
- The audit trail (`decisions_made`) contains only completed runs. Abandoned runs no longer happen.
- Cost: held messages aren't processed by the LLM until the underwriter asks for a re-screen.
- Risk: the review queue must not rely on interrupt markers alone (see the gotcha above).
