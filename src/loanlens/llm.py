"""The one place LoanLens creates its chat model.

Every LangGraph node gets its model from `get_model()`, so switching provider
(Foundry -> OpenAI -> anything OpenAI-compatible) is a change in this file only.

Uses the Azure OpenAI **v1 API** exposed by a Microsoft Foundry resource:
plain `ChatOpenAI` pointed at `https://<resource>.openai.azure.com/openai/v1/`.
No api_version needed on v1.
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
    load_dotenv()
    return ChatOpenAI(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],  # on Azure, 'model' = YOUR deployment name
        base_url=v1_base_url(os.environ["AZURE_OPENAI_ENDPOINT"]),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        **kwargs,
    )
