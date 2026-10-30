"""Lesson 0 smoke test: can LoanLens talk to the model deployed in your Microsoft Foundry resource?

Run:  uv run python scripts/smoke_test.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from loanlens.llm import REQUIRED_ENV, get_model, v1_base_url  # noqa: E402

HINTS = {
    "401": "Key rejected. In Foundry, re-copy the API key into AZURE_OPENAI_API_KEY.",
    "DeploymentNotFound": "Deployment not found. AZURE_OPENAI_DEPLOYMENT_NAME must be the name YOU gave the deployment, not the model name.",
    "404": "Not found. Check the endpoint is https://<resource>.openai.azure.com/openai/v1/ and the deployment name is exact (case matters).",
    "Connection error": "Can't reach the endpoint. Check AZURE_OPENAI_ENDPOINT for typos (and your internet connection).",
    "429": "Rate limited / quota. Wait a minute, or raise the deployment's tokens-per-minute limit in Foundry.",
}


def main() -> int:
    load_dotenv()
    missing = [k for k in REQUIRED_ENV if not os.getenv(k) or "<" in os.getenv(k, "")]
    if missing:
        print("❌ Missing or placeholder values in .env:", ", ".join(missing))
        print("   Copy .env.example to .env and fill them in.")
        return 1

    print("Endpoint:  ", v1_base_url(os.environ["AZURE_OPENAI_ENDPOINT"]))
    print("Deployment:", os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"])
    try:
        reply = get_model().invoke("Reply with exactly the word: READY")
    except Exception as exc:  # friendly hint for the common setup errors
        text = str(exc)
        print("❌ Call failed:", text[:300])
        for needle, hint in HINTS.items():
            if needle in text:
                print("💡", hint)
                break
        return 1

    usage = getattr(reply, "usage_metadata", None) or {}
    print("Model said:", str(reply.content).strip())
    print("Tokens used:", usage.get("total_tokens", "n/a"))
    print("✅ LoanLens can talk to your Foundry model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
