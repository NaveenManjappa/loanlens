"""Lesson 0 smoke test: can LoanLens talk to Azure OpenAI?

Run:  uv run python scripts/smoke_test.py
"""

import os
import sys

from dotenv import load_dotenv

REQUIRED = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
]

HINTS = {
    "401": "Key rejected. Re-copy Key 1 from 'Keys and Endpoint' into AZURE_OPENAI_API_KEY.",
    "DeploymentNotFound": "Deployment not found. AZURE_OPENAI_DEPLOYMENT_NAME must be the name YOU gave the deployment, not the model name.",
    "404": "Not found. Check the endpoint URL (https://<resource>.openai.azure.com/) and the deployment name.",
    "api-version": "API version rejected. Try the version shown in your deployment's code sample in the portal.",
    "Connection error": "Can't reach the endpoint. Check AZURE_OPENAI_ENDPOINT for typos (and your internet connection).",
    "getaddrinfo": "Can't reach the endpoint. Check AZURE_OPENAI_ENDPOINT for typos.",
    "429": "Rate limited / quota. Wait a minute, or raise the deployment's tokens-per-minute limit.",
}


def main() -> int:
    load_dotenv()
    missing = [k for k in REQUIRED if not os.getenv(k) or "<" in os.getenv(k, "")]
    if missing:
        print("❌ Missing or placeholder values in .env:", ", ".join(missing))
        print("   Copy .env.example to .env and fill them in.")
        return 1

    from langchain.chat_models import init_chat_model

    model = init_chat_model(
        "azure_openai:" + os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    )
    try:
        reply = model.invoke("Reply with exactly the word: READY")
    except Exception as exc:  # show a friendly hint for the common setup errors
        text = str(exc)
        print("❌ Call failed:", text[:300])
        for needle, hint in HINTS.items():
            if needle in text:
                print("💡", hint)
                break
        return 1

    usage = getattr(reply, "usage_metadata", None) or {}
    print("Model said:", reply.content.strip())
    print("Tokens used:", usage.get("total_tokens", "n/a"))
    print("✅ LoanLens can talk to Azure OpenAI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
