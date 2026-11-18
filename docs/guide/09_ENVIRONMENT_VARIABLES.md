# 🔑 Environment Variables and Configuration

All configuration comes from environment variables, so the same Docker image runs locally and in Azure. For
local development, copy `.env.example` to `.env` and fill it in. **Never commit `.env`.**

LoanLens's own settings are declared once, as typed fields on `Settings` in `config/settings.py`
(`pydantic-settings`). The variable name is the field name in upper case. Values in the real environment win
over `.env`.

---

## 🧠 Azure OpenAI (required to run the app)

| Variable | Description | Example |
| :--- | :--- | :--- |
| `AZURE_OPENAI_ENDPOINT` | Your Microsoft Foundry resource. The bare URL or the full v1 URL both work; `v1_base_url()` normalises it | `https://my-resource.openai.azure.com/openai/v1/` |
| `AZURE_OPENAI_API_KEY` | The resource's API key. Held as a `SecretStr`, so it is hidden when printed | `abc123...` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | The name **you** gave the deployment in Foundry, **not** the model name | `gpt-5-loanlens` |

These are optional on `Settings` so tests can build it without them. `get_model()` raises a clear
`RuntimeError` naming every missing variable.

---

## 🗄️ Storage

| Variable | Description | Default |
| :--- | :--- | :--- |
| `DATABASE_URL` | Postgres connection string. When set, cases and locks use Postgres. A `SecretStr` | unset → SQLite |
| `SQLITE_PATH` | The SQLite file used when `DATABASE_URL` is unset | `loanlens_cases.db` |

---

## 📜 Logging

| Variable | Description | Default |
| :--- | :--- | :--- |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR` (lower case accepted) | `INFO` |

---

## 🕵️ LangSmith tracing (optional)

These are read by the **LangSmith library itself**, not by `Settings`. `main.py` calls `load_dotenv()` so values
in `.env` reach `os.environ`, where LangSmith looks for them.

| Variable | Description | Example |
| :--- | :--- | :--- |
| `LANGSMITH_TRACING` | Turn tracing on | `true` |
| `LANGSMITH_API_KEY` | From smith.langchain.com → Settings → API Keys | `lsv2_...` |
| `LANGSMITH_PROJECT` | Project-based workspace: the project to trace into | `loanlens` |
| `LANGSMITH_AGENT_ID` / `LANGSMITH_AGENT_ENVIRONMENT` | Agent-based workspace: use these **instead of** `LANGSMITH_PROJECT` | |
| `LANGSMITH_HIDE_INPUTS` | Keep model inputs (customer data) out of traces. **Keep this `true`** | `true` |
| `LANGSMITH_ENDPOINT` | Only for an EU-region account | `https://eu.api.smith.langchain.com` |

> Comments in `.env` go on their own line: `docker --env-file` doesn't strip inline comments.

---

## 🧪 Test-only switches

| Variable | Effect |
| :--- | :--- |
| `LOANLENS_LIVE=1` | Run the tests marked `@pytest.mark.live`, which call the real model |
| `LOANLENS_PG_URL` | Run the Postgres storage and lock tests against this database |

---

## ☁️ In Azure

Secrets are stored on the container app and referenced with `secretref:`, so they're never in the image or in
plain text in the app's configuration:

```bash
--secrets aoai-key="$AZURE_OPENAI_API_KEY" database-url="$DATABASE_URL" \
--env-vars AZURE_OPENAI_API_KEY=secretref:aoai-key DATABASE_URL=secretref:database-url ...
```

See [13](13_RUNNING_AND_DEPLOYING.md) and the full [Azure runbook](../deploy-azure.md).

---

## ➕ Adding a setting

1. Add a typed field to `Settings` in `config/settings.py` (use `SecretStr` for anything secret).
2. Read it via `get_settings()` (or receive `Settings` as an argument), never `os.getenv`.
3. Add it to `.env.example` and to this page.
