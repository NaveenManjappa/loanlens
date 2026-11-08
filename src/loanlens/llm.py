"""The one place LoanLens creates its chat model, so switching provider is a change in this file only.

Uses the Azure OpenAI v1 API exposed by a Microsoft Foundry resource: plain ChatOpenAI pointed at
https://<resource>.openai.azure.com/openai/v1/ (no api_version needed on v1).
Docs: https://docs.langchain.com/oss/python/integrations/providers/microsoft#azure-openai
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

REQUIRED_ENV = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]


def v1_base_url(endpoint: str) -> str:
    """Accept either the bare resource URL or the full v1 URL; always return the v1 URL."""
    url = endpoint.strip().rstrip("/")
    if not url.endswith("/openai/v1"):
        url += "/openai/v1"
    return url + "/"


def get_model(**kwargs) -> ChatOpenAI:
    """The configured chat model. kwargs are passed to ChatOpenAI (e.g. temperature)."""
    load_dotenv()
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")

    return ChatOpenAI(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],  # on Azure, 'model' is YOUR deployment name
        base_url=v1_base_url(os.environ["AZURE_OPENAI_ENDPOINT"]),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        **kwargs,
    )
