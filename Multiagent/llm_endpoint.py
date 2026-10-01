from __future__ import annotations

import logging
import os
from pathlib import Path

import dotenv
import httpx
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

DEFAULT_ENV_PATH = Path(__file__).parent / ".env"

DEFAULT_MODEL = "gpt-4o"


def tls_verify() -> bool | str:
    """
    Determines the TLS verification for the LLM endpoint.

    The private cluster uses a self-signed certificate. ``LLM_CA_BUNDLE`` pins exactly that
    certificate as trust anchor, which keeps protection against man-in-the-middle; a
    relative path is resolved against the project root. ``LLM_VERIFY_SSL=false`` disables
    verification entirely and is only an emergency fallback.

    :return: Path of the CA bundle, ``False`` if verification is disabled, else ``True``.
    :raises FileNotFoundError: If ``LLM_CA_BUNDLE`` points to no file.
    """
    bundle = os.getenv("LLM_CA_BUNDLE")
    if bundle:
        path = Path(bundle)
        if not path.is_absolute():
            path = Path(__file__).parents[1] / bundle
        if not path.is_file():
            raise FileNotFoundError(f"LLM_CA_BUNDLE does not point to a file: {path}")
        return str(path)
    if (os.getenv("LLM_VERIFY_SSL") or "").strip().lower() in {"false", "0", "no"}:
        logger.warning(
            "TLS verification for the LLM endpoint is disabled (LLM_VERIFY_SSL). "
            "Prefer setting LLM_CA_BUNDLE to the server certificate."
        )
        return False
    return True


class _GatewayChatOpenAI(ChatOpenAI):
    """
    ``ChatOpenAI`` adapted to the OpenWebUI gateway of the cluster.

    LangChain sends ``content: null`` for assistant messages with ``tool_calls``, which the
    gateway rejects with ``400 - object of type 'NoneType' has no len()`` as soon as a tool
    result is passed back; every agent with tools would fail. Replacing ``null`` with ``""``
    is equivalent for the OpenAI API.
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        """
        Builds the request body and replaces ``content: null`` with an empty string.

        :param input_: Messages passed to the model.
        :param stop: Optional stop sequences.
        :param kwargs: Further arguments passed to the parent implementation.
        :return: The request payload for the endpoint.
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload.get("messages") or []:
            if isinstance(message, dict) and message.get("content") is None:
                message["content"] = ""
        return payload


def use_openai_default() -> bool:
    """
    Checks whether ``SUPERVISOR_USE_OPENAI`` in the .env routes the chat to api.openai.com.

    :return: ``True`` if the switch is set.
    """
    return (os.getenv("SUPERVISOR_USE_OPENAI") or "").strip().lower() in {"true", "1", "yes"}


def build_chat_openai(
    *,
    model: str | None = None,
    temperature: float | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    default_model: str = DEFAULT_MODEL,
    use_openai: bool = False,
    env_path: str | Path | None = None,
) -> ChatOpenAI:
    """
    Builds a ``ChatOpenAI`` client for the configured endpoint, shared by all components.

    Each value is resolved as argument, then .env (``LLM_BASE_URL``, ``LLM_MODEL``,
    ``LLM_API_KEY``), then default. ``LLM_API_KEY`` is only used for a custom endpoint, so the
    OpenAI key never leaks to a foreign endpoint. Custom httpx clients and the gateway fix
    are only applied to a custom endpoint.

    :param model: Model name; falls back to the configured default.
    :param temperature: Sampling temperature; the LangChain default when omitted.
    :param base_url: Endpoint URL; taken from ``LLM_BASE_URL`` when omitted.
    :param api_key: API key; taken from ``LLM_API_KEY`` when omitted.
    :param default_model: Model used when neither argument nor environment supplies one.
    :param use_openai: ``True`` routes the call to api.openai.com with ``OPENAI_API_KEY``,
        for the model comparison.
    :param env_path: Path of the ``.env`` to read.
    :return: The configured ``ChatOpenAI`` client.
    """
    dotenv.load_dotenv(env_path or DEFAULT_ENV_PATH)

    if use_openai:
        resolved_base = None
        resolved_model = model or default_model
        key = api_key
    else:
        resolved_base = base_url or os.getenv("LLM_BASE_URL") or None
        resolved_model = model or os.getenv("LLM_MODEL") or default_model
        key = api_key or (os.getenv("LLM_API_KEY") if resolved_base else None)

    verify = tls_verify() if resolved_base else True
    tls: dict = {}
    if verify is not True:
        tls = {
            "http_client": httpx.Client(verify=verify),
            "http_async_client": httpx.AsyncClient(verify=verify),
        }

    cls = _GatewayChatOpenAI if resolved_base else ChatOpenAI

    llm = cls(
        model=resolved_model,
        **({"temperature": temperature} if temperature is not None else {}),
        **({"base_url": resolved_base} if resolved_base else {}),
        **({"api_key": key} if key else {}),
        **tls,
    )
    logger.info(
        "LLM: model '%s' via %s", resolved_model, resolved_base or "api.openai.com"
    )
    return llm
