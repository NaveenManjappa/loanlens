"""The model provider. The one place LoanLens creates its chat model, so switching provider is a
change in this file only. Also recognises the provider's content-filter errors (ADR 0002).

Uses the Azure OpenAI v1 API exposed by a Microsoft Foundry resource: plain ChatOpenAI pointed at
https://<resource>.openai.azure.com/openai/v1/ (no api_version needed on v1).
Docs: https://docs.langchain.com/oss/python/integrations/providers/microsoft#azure-openai
"""

import openai
from langchain_openai import ChatOpenAI

from loanlens.config.settings import Settings, get_settings

REQUIRED_ENV = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]


def v1_base_url(endpoint: str) -> str:
    """Accept either the bare resource URL or the full v1 URL; always return the v1 URL."""
    url = endpoint.strip().rstrip("/")
    if not url.endswith("/openai/v1"):
        url += "/openai/v1"
    return url + "/"


def get_model(settings: Settings | None = None, **kwargs) -> ChatOpenAI:
    """The configured chat model. kwargs are passed to ChatOpenAI (e.g. temperature)."""
    settings = settings or get_settings()
    endpoint = settings.azure_openai_endpoint
    api_key = settings.azure_openai_api_key
    deployment = settings.azure_openai_deployment_name
    if not (endpoint and api_key and deployment):
        missing = [
            name
            for name, value in zip(REQUIRED_ENV, (endpoint, api_key, deployment), strict=True)
            if not value
        ]
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)} (see .env.example)")

    return ChatOpenAI(
        model=deployment,  # on Azure, 'model' is YOUR deployment name
        base_url=v1_base_url(endpoint),
        api_key=api_key,
        **kwargs,
    )


def is_content_filter_error(exc: BaseException) -> bool:
    """True if exc means the provider's content filter refused the request.

    That's a blocked prompt (an error whose .code is "content_filter": Azure's HTTP 400) or a
    blocked completion (openai.ContentFilterFinishReasonError).
    """
    blocked_prompt = getattr(exc, "code", None) == "content_filter"
    blocked_completion = isinstance(exc, openai.ContentFilterFinishReasonError)
    return blocked_prompt or blocked_completion
