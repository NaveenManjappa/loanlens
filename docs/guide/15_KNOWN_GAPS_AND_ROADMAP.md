# 🗺️ Known Gaps and Roadmap

LoanLens is a portfolio system, not a production lender. This page lists what is deliberately not done yet, and
where each item is designed.

---

## 🔴 Before any real customer

| Gap | Risk | Plan |
| :--- | :--- | :--- |
| **No authentication** | Anyone with the URL can read any case, and decide as any `underwriter_id` | Microsoft Entra ID through Container Apps' built-in authentication, with the underwriter's identity taken from the login, not the request body. [ADR 0004](../decisions/0004-authentication-entra-id.md) |
| **No audit log** | Decisions live only in checkpoints, which aren't append-only or access-controlled | An append-only, access-controlled audit store; blocked messages kept there under retention rules (ADR 0002) |
| **Automated declines** | Under-18s are declined with no human | It's a legal-capacity rule, but production still needs Article 22 safeguards: telling the applicant, and human review on request |
| **Vulnerable customers (Consumer Duty)** | Vulnerability markers are checked only on referred cases; the content filter is not a vulnerability detector | Vulnerability detection on every case |
| **Fake external systems** | Credit bureau, HMRC income and sanctions data are canned | Real clients with timeouts and retries in `integrations/`, injected into the tools |

## 🟠 Reliability

| Gap | Plan |
| :--- | :--- |
| **No idempotency keys**: a retried request runs the graph again | `Idempotency-Key` header; replay the first response. [ADR 0003](../decisions/0003-idempotency-keys.md) |
| **No explicit model timeout**: relies on the OpenAI client's default | A short timeout and bounded retries as settings |
| **Schema setup at startup** (`saver.setup()`) | Run it as a separate migration step, so replicas never race to change the database |
| **Employment status from a job title** is a prompt rule only | A code guard like `ground_age` |

## 🟡 Product

| Gap | Plan |
| :--- | :--- |
| **No UI** | An applicant chat and an underwriter review queue, backed by a small `cases` table. [ADR 0005](../decisions/0005-ui-and-review-queue.md) |
| **No way to list cases** | The same `cases` table |
| **No re-screening** after the applicant sends new facts during review | A `rescreen` decision that loops back to `start_run` and always returns to a human. [ADR 0001](../decisions/0001-underwriter-rescreen.md) part 2 |

## 🔵 Platform and engineering

| Gap | Plan |
| :--- | :--- |
| Model key stored as a secret | Managed identity to Azure OpenAI, no key at all |
| Case data with a second provider (Neon) | A data-processing agreement, or Azure-hosted Postgres with private networking |
| Manual deployment | A deploy job using GitHub's federated sign-in to Azure (OIDC) |
| No alerts | Alerts on error rates and eval scores |
| No type checker in CI | Add pyright in standard mode to `make check` and CI |

---

## 📚 Decision records

| ADR | Title | Status |
| :--- | :--- | :--- |
| [0001](../decisions/0001-underwriter-rescreen.md) | Messages during underwriter review, and re-screening | Part 1 built, part 2 designed |
| [0002](../decisions/0002-content-safety-provider-filter.md) | Harmful and off-topic input: rely on the provider's content filter | Accepted, error handling built |
| [0003](../decisions/0003-idempotency-keys.md) | Idempotency keys for requests that change a case | Proposed |
| [0004](../decisions/0004-authentication-entra-id.md) | Authentication with Microsoft Entra ID | Proposed |
| [0005](../decisions/0005-ui-and-review-queue.md) | A two-screen UI and a review queue | Proposed |
