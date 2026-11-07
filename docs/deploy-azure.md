# Deploying LoanLens to Azure Container Apps

A runbook for Lesson 12. It deploys the API with a Postgres checkpointer, so cases survive
restarts, redeploys and multiple replicas.

```
 browser / curl ──HTTPS──> Azure Container Apps (loanlens-api, 0..2 replicas)
                                 │                     │
                                 ▼                     ▼
                     Azure Database for         Azure OpenAI
                     PostgreSQL (checkpoints)   (your Foundry deployment)
```

> **Demo only.** The API has no authentication yet. Anyone with the URL can read a case
> and submit an underwriter decision. Use made-up applicants only, and delete everything when
> you're done (step 7).

## 0. One-time setup (on your Mac)

```bash
brew install azure-cli
az login
az extension add --name containerapp --upgrade --allow-preview true
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.DBforPostgreSQL
```

## 1. Names and secrets for this shell

Run these from the `loanlens/` folder:

```bash
set -a; source .env; set +a                 # loads your AZURE_OPENAI_* values
RG=rg-loanlens
LOC=uksouth
APP=loanlens-api
ENV_NAME=loanlens-env
PG=loanlens-pg-$RANDOM                      # must be globally unique
PG_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=')    # letters and digits only: safe in a URL
echo "Postgres server: $PG"                 # note it down
```

## 2. Resource group and Postgres (about 5–10 minutes)

```bash
az group create --name $RG --location $LOC

az postgres flexible-server create \
  --resource-group $RG --name $PG --location $LOC \
  --tier Burstable --sku-name Standard_B1ms --storage-size 32 --version 16 \
  --admin-user loanlens --admin-password "$PG_PASSWORD" \
  --public-access 0.0.0.0

DATABASE_URL="postgresql://loanlens:${PG_PASSWORD}@${PG}.postgres.database.azure.com:5432/postgres?sslmode=require"
```

- `--public-access 0.0.0.0` lets **Azure services** connect, and the password and TLS
  (`sslmode=require`) protect the server. In production you'd use private networking instead.
- Optional check before deploying: run
  `LOANLENS_PG_URL="$DATABASE_URL" uv run pytest tests/test_lesson12_checkpointing.py -v`.
  This test connects from your Mac, so it needs a firewall rule for your IP first:
  `az postgres flexible-server firewall-rule create -g $RG -n $PG --rule-name me --start-ip-address <your IP> --end-ip-address <your IP>`.

## 3. Build and deploy the app (about 5 minutes)

```bash
az containerapp up \
  --name $APP --resource-group $RG --location $LOC \
  --environment $ENV_NAME --source . --ingress external
```

This creates a container registry and a Container Apps environment, builds the `Dockerfile`, and
deploys the image. **The first revision will fail to start, and that is expected**: the app needs
its settings, which you add in the next step.

## 4. Settings and secrets

```bash
az containerapp secret set --name $APP --resource-group $RG \
  --secrets aoai-key="$AZURE_OPENAI_API_KEY" database-url="$DATABASE_URL"

az containerapp update --name $APP --resource-group $RG \
  --min-replicas 0 --max-replicas 2 \
  --set-env-vars \
    AZURE_OPENAI_ENDPOINT="$AZURE_OPENAI_ENDPOINT" \
    AZURE_OPENAI_DEPLOYMENT_NAME="$AZURE_OPENAI_DEPLOYMENT_NAME" \
    AZURE_OPENAI_API_KEY=secretref:aoai-key \
    DATABASE_URL=secretref:database-url
```

`secretref:` means the value is stored as a **secret** on the container app. It's injected at
runtime and doesn't appear in plain text in the app's configuration or in the image.

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

FastAPI also serves interactive docs at `$URL/docs`.

**The durability test:** restart the app, then read the case again. It's still there, because it
lives in Postgres, not in the container:

```bash
az containerapp revision restart -n $APP -g $RG \
  --revision $(az containerapp revision list -n $APP -g $RG --query "[0].name" -o tsv)
curl $URL/cases/demo-1
```

Logs: `az containerapp logs show -n $APP -g $RG --follow`

## 6. Redeploy after a code change

Run the step 3 command again with the same names. The settings and secrets from step 4 are kept.

## 7. Clean up (do this when you're done)

```bash
az group delete --name $RG --yes --no-wait
```

This deletes the Postgres server, the registry, the environment and the app.

## Cost (approximate, check the Azure pricing pages)

| Resource | Cost while it exists |
|---|---|
| Postgres Flexible Server, Burstable B1ms | about $14/month compute, plus storage. **The main cost** |
| Container Registry (Basic) | a few dollars a month |
| Container Apps (Consumption) | a free monthly grant (180,000 vCPU-seconds, 2M requests); with 0 minimum replicas, nothing is charged while idle |
| Azure OpenAI | per token, as before |

Scale-to-zero means the first request after a quiet spell is slow (a cold start of a few seconds).

## Not production-ready yet (known gaps)

- **No authentication.** Applicants need sessions, and the underwriter endpoints need Microsoft
  Entra ID, for example Container Apps' built-in authentication.
- **No protection against concurrent messages on the same case.** Two requests at once can race.
- **The API key is stored as a secret.** In production, use a managed identity to reach Azure OpenAI
  with no key at all.
- **No audit log or vulnerability detection** (see ADR 0002).
- **Postgres is on public access.** In production, use private networking.
- **No CI/CD and no monitoring alerts.**
