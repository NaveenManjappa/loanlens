# Deploying LoanLens to Azure Container Apps (near-zero cost)

A runbook for Lesson 12. The API runs on **Azure Container Apps**. The image is built by
**GitHub Actions** and stored on **GitHub Container Registry** (ghcr.io). The cases live in a free
**Neon** Postgres database in London. Traces go to **LangSmith**.

```
 git push ──▶ GitHub Actions ──tests pass──▶ ghcr.io/naveenmanjappa/loanlens  (the IMAGE)
                                                       │ pulled when a replica starts
                                                       ▼
                                      Azure Container Apps: FastAPI + LoanLens  (your HTTPS URL)
                                         │             │              │            │
                                         ▼             ▼              ▼            ▼
                                   Neon Postgres   Azure OpenAI    LangSmith    Azure logs
                                   (the CASES)     (the model)     (traces)     (console output)
```

> **Demo only.** The API has no authentication yet. Anyone with the URL can read a case
> and submit an underwriter decision. Use made-up applicants only.

## 0. One-time setup (on your Mac)

```bash
brew install azure-cli
az login
az extension add --name containerapp --upgrade --allow-preview true
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

## 1. The database: Neon (about 5 minutes, free)

1. Sign up at [neon.com](https://neon.com) and create a project. Choose the region
   **AWS Europe (London)**, `aws-eu-west-2`, so case data stays in the UK and close to `uksouth`.
2. Copy the **direct** connection string (not the "pooled" one). It looks like
   `postgresql://user:password@ep-xxxx.eu-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require`.
3. Optional check from your Mac. A case should survive a "restart":
   ```bash
   LOANLENS_PG_URL='<the connection string>' uv run pytest tests/test_lesson12_checkpointing.py -v
   ```

The free plan suspends the database after 5 minutes without traffic. The next query wakes it in
about a second.

## 2. The image: GitHub Actions publishes it to ghcr.io

The workflow `.github/workflows/publish-image.yml` runs on every push to `main`:

1. **test**: `uv sync --frozen`, then `uv run pytest`. Only fake models run, so no secrets are needed.
2. **publish**: only if the tests pass, it builds the `Dockerfile` for `linux/amd64` and pushes
   `ghcr.io/naveenmanjappa/loanlens:<commit sha>` and `:latest`.

To use it:

```bash
git add -A && git commit -m "Lesson 12: API, Docker, deploy workflow" && git push
```

Then open the repo on GitHub → **Actions** and watch the run. When it's green, the image appears
under your profile → **Packages** → `loanlens`.

- **The package must be public**, so Azure can pull it without a password. A package published by a
  workflow takes the repository's visibility, so a public repo gives you a public package. If it's
  private: open the package → **Package settings** → **Change visibility** → Public.
- A public image is safe only because it holds **no secrets**. That is the job of `.dockerignore`
  (no `.env`, no `*.db`). Secrets go into Azure in step 4.
- `uv.lock` must be committed: `--frozen` installs exactly what it lists.

## 3. Names and secrets for this shell

Run these from the `loanlens/` folder:

```bash
set -a; source .env; set +a        # loads AZURE_OPENAI_* (and LANGSMITH_API_KEY if it's in .env)
RG=rg-loanlens
LOC=uksouth
APP=loanlens-api
ENV_NAME=loanlens-env
IMAGE=ghcr.io/naveenmanjappa/loanlens
DATABASE_URL='<the Neon connection string>'   # single quotes, no spaces around =
```

## 4. Create the app (about 5 minutes)

```bash
az group create --name $RG --location $LOC
az containerapp env create --name $ENV_NAME --resource-group $RG --location $LOC

az containerapp create \
  --name $APP --resource-group $RG --environment $ENV_NAME \
  --image $IMAGE:$(git rev-parse HEAD) \
  --ingress external --target-port 8000 \
  --min-replicas 0 --max-replicas 2 \
  --secrets aoai-key="$AZURE_OPENAI_API_KEY" database-url="$DATABASE_URL" \
  --env-vars \
    AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
    AZURE_OPENAI_DEPLOYMENT_NAME="$AZURE_OPENAI_DEPLOYMENT_NAME" \
    AZURE_OPENAI_API_KEY=secretref:aoai-key \
    DATABASE_URL=secretref:database-url
```

- `--image $IMAGE:$(git rev-parse HEAD)` deploys **exactly the commit you pushed**. Only use it after
  that commit's workflow run is green.
- `secretref:` means the value is stored as a **secret** on the container app and injected at
  runtime. It's never in the image, and it doesn't appear in plain text in the app's configuration.

**Optional: LangSmith tracing.** Add `LANGSMITH_API_KEY` to `.env` (see `.env.example`), run
`set -a; source .env; set +a` again, then:

```bash
az containerapp secret set -n $APP -g $RG --secrets langsmith-key="$LANGSMITH_API_KEY"
az containerapp update -n $APP -g $RG --set-env-vars \
  LANGSMITH_TRACING=true LANGSMITH_API_KEY=secretref:langsmith-key \
  LANGSMITH_PROJECT=loanlens LANGSMITH_HIDE_INPUTS=true
```

Use the same workspace variables as in Lesson 11. If your workspace is agent-based, replace
`LANGSMITH_PROJECT` with `LANGSMITH_AGENT_ID` and `LANGSMITH_AGENT_ENVIRONMENT`. If your LangSmith
account is in the EU region, also add `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com`.
`LANGSMITH_HIDE_INPUTS=true` keeps model inputs out of the traces (Lesson 7's leak path 4).

## 5. Try it

```bash
URL=https://$(az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv)

curl $URL/health
curl -X POST $URL/cases/demo-1/messages -H 'content-type: application/json' \
  -d '{"text": "Sam Jones, 29, self-employed web designer, about £40k a year, want £5,000"}'
curl $URL/cases/demo-1/review
curl -X POST $URL/cases/demo-1/decision -H 'content-type: application/json' \
  -d '{"decision": "approve", "reason": "Two years of accounts seen", "underwriter_id": "uw-1"}'
```

FastAPI also serves interactive docs at `$URL/docs`. The first request after a quiet spell can take
several seconds, because both the app and Neon are waking from zero.

**The durability test:** restart the app, then read the case again. It's still there, because it
lives in Neon, not in the container:

```bash
az containerapp revision restart -n $APP -g $RG \
  --revision $(az containerapp revision list -n $APP -g $RG --query "[0].name" -o tsv)
curl $URL/cases/demo-1
```

## 6. Logs and traces

- **Azure: what the app prints** (requests, startup, errors):
  - live in the terminal: `az containerapp logs show -n $APP -g $RG --follow`
  - in the portal: the container app → **Monitoring** → *Log stream* (live) or *Logs* (searchable history)
- **LangSmith: what happens inside the graph** (nodes, model calls, tokens, timings). Look in the
  `loanlens` project.
- **Neon: database activity and storage**, in the Neon console.

## 7. Redeploy after a code change

```bash
git push                                   # wait for the workflow run to go green
az containerapp update -n $APP -g $RG --image $IMAGE:$(git rev-parse HEAD)
```

Settings and secrets are kept. Each update creates a new **revision**, so you can always see which
commit is live.

## 8. Clean up

```bash
az group delete --name $RG --yes --no-wait   # the app, its environment and logs
```

The Neon project and the ghcr.io image are free to keep. Delete them in their own consoles if you
want to.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| The workflow stops at `test` | A test fails, or `uv.lock` wasn't committed after `uv add`. Run `uv run pytest` locally, then commit the lock file. |
| `az containerapp create` can't pull the image | The package is private, or the workflow for that commit hasn't finished (step 2). |
| `/health` never answers | The app crashes at startup. Run `az containerapp logs show -n $APP -g $RG --type system`, and the console logs without `--type`. `KeyError: 'AZURE_OPENAI_…'` means a setting is missing. |
| 503 "temporarily unavailable" on messages | The model call failed: check the endpoint, deployment name and key settings. |
| Errors connecting to the database | Use the *direct* Neon string (no `-pooler`), including `sslmode=require`, in single quotes. |

## Cost

| Resource | Cost |
|---|---|
| Container Apps (Consumption, 0 minimum replicas) | free monthly grant (180,000 vCPU-seconds, 2M requests); nothing while idle |
| Log Analytics (created with the environment) | a demo stays within the small free allowance |
| ghcr.io (public image) and GitHub Actions (public repo) | free |
| Neon Free plan | free: 100 compute-hours per project per month, 1 GB storage per project |
| Azure OpenAI | per token, as before |

## Not production-ready yet (known gaps)

- **No authentication.** Applicants need sessions, and the underwriter endpoints need Microsoft
  Entra ID, for example Container Apps' built-in authentication.
- **Concurrent requests are rejected, not deduplicated.** A per-case lock (a Postgres advisory lock) stops two
  requests changing one case at once: the second gets 409. An idempotency key would let a retry get the
  first answer instead.
- **The model key is stored as a secret.** In production, use a managed identity to reach Azure
  OpenAI with no key at all.
- **The case data sits with a second provider** (Neon). It's in London, but production would need a
  data-processing agreement, or Azure-hosted Postgres with private networking.
- **Deployment is manual.** Next steps would be a deploy job using GitHub's federated sign-in to
  Azure (OIDC, no stored Azure password), and alerts on error rates and eval scores.
- **No audit log or vulnerability detection** (see ADR 0002).
