# ADR 0004: Authentication with Microsoft Entra ID, identity from the login

**Status:** Proposed. Not built.
**Date:** November 2026

## Context

The API has no authentication. That causes three concrete problems:

1. **Anyone can act as an underwriter.** `POST /cases/{case_id}/decision` takes `underwriter_id` from the
   request body, so a caller can approve a loan in someone else's name. The audit trail records a name,
   but nothing proves it.
2. **Anyone can read any case.** Case ids are chosen by the caller (`demo-1`), so they're guessable, and
   `GET /cases/{case_id}` and `GET /cases/{case_id}/review` return case data to whoever asks.
3. **Significant decisions need a real, accountable person.** Human review is LoanLens's answer to UK GDPR
   Article 22 concerns, and it's only meaningful if the reviewer's identity can be trusted.

LoanLens runs on Azure Container Apps, which has built-in authentication: it can sign users in with
Microsoft Entra ID before a request reaches the app, and pass the user's identity in request headers such
as `X-MS-CLIENT-PRINCIPAL-ID` and `X-MS-CLIENT-PRINCIPAL-NAME`. Microsoft's documentation states that
external requests aren't allowed to set these headers, so they're present only when Container Apps sets
them.

## Options considered

1. **Container Apps built-in authentication with Entra ID (chosen).** Azure handles the sign-in, token
   validation and sessions; the app reads the identity headers. Very little code, and the standard choice
   for a UK enterprise or financial-services Azure estate.
2. **Validate Entra ID tokens in FastAPI** (for example with MSAL or a JWT library). More code and more to
   get right (keys, expiry, audience), but it works on any host. Worth it only if LoanLens leaves Azure.
3. **API keys.** Rejected: a shared key identifies a client, not a person, so it can't support a
   trustworthy audit trail.

## Decision

**Phase 1: underwriters.**
- Turn on built-in authentication with Microsoft Entra ID, set to **allow unauthenticated requests**, so
  that applicant routes stay open and the app itself decides which routes need a login. Microsoft notes
  that "require authentication" applies to every request, which would also lock out applicants.
- An Entra ID app role **`Underwriter`**. The review and decision routes (`GET /reviews`,
  `GET /cases/{case_id}/review`, `POST /cases/{case_id}/decision`) require a signed-in user with that role:
  `401` if not signed in, `403` without the role.
- **The underwriter's identity comes from the login, never from the request body.** `underwriter_id` is
  removed from `DecisionIn`, and the decision records the user's Entra object id and name from the headers.
- A FastAPI dependency (`current_underwriter`) reads and checks the headers. Locally, where there's no
  Container Apps, tests and development override that dependency. The override must be impossible to
  switch on in the deployed app.

**Phase 1: applicants.**
- The server creates case ids: a new `POST /cases` returns a random UUID, and callers can no longer choose
  ids. The id works as a hard-to-guess reference for that one case, which is enough for a demo.
- Applicant routes get basic rate limiting, so ids can't be guessed by brute force.

**Phase 2 (later): applicant accounts** with Microsoft Entra External ID (customer sign-in), so an
applicant can only ever see their own cases.

## Consequences

- Decisions carry a verified identity, which makes the audit trail trustworthy.
- Underwriter routes can't be called anonymously or with a made-up name.
- The authentication layer depends on Azure Container Apps. Moving host means adopting option 2.
- Tests must simulate the identity headers (one place, the dependency override).
- Until phase 2, anyone holding a case id can read that case: acceptable for a demo, not for production.
- Signing in needs an Entra ID app registration, and app-role assignments managed in Azure.

## Revisit when

- LoanLens takes real applicants (do phase 2 first);
- it moves off Azure Container Apps (validate tokens in the app instead);
- underwriters need finer permissions, for example approval limits by loan amount.

## Sources

- [Microsoft Learn: Authentication and authorization in Azure Container Apps](https://learn.microsoft.com/azure/container-apps/authentication)
