"""LiteLLM integration — cost tracking for every model LiteLLM routes.

Works with both the LiteLLM SDK and proxy callbacks. No litellm import is
required: LiteLLM accepts plain callables in ``success_callback`` /
``failure_callback``, so this module stays dependency-free.

Usage::

    import litellm
    from stepcost import StepCost
    from stepcost.integrations.litellm import StepCostLiteLLMLogger

    cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")
    logger = StepCostLiteLLMLogger(cc)
    litellm.success_callback = [logger.log_success_event]
    litellm.failure_callback = [logger.log_failure_event]

    with cc.trace(feature_id="chat", customer_id="acme") as trace:
        litellm.completion(model="claude-haiku-4-5", messages=[...])

    print(trace.total_usd)
"""

from __future__ import annotations

from typing import Any

from stepcost.client import StepCost
from stepcost.models import Provider, SpanKind, TokenUsage


def _provider_from(kwargs: dict, model: str) -> Provider:
    lp = kwargs.get("litellm_params") or {}
    hint = str(lp.get("custom_llm_provider") or "").lower()
    hay = f"{hint} {model.lower()}"
    if "anthropic" in hay or "claude" in hay:
        return Provider.ANTHROPIC
    if "bedrock" in hay:
        return Provider.BEDROCK
    if "vertex" in hay or "gemini" in hay:
        return Provider.VERTEX
    if "ollama" in hay:
        return Provider.OLLAMA
    if "openai" in hay or "gpt" in hay or model.lower().startswith("o"):
        return Provider.OPENAI
    return Provider.OTHER


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _usage_from_response(response_obj: Any) -> TokenUsage | None:
    usage = _attr(response_obj, "usage")
    if usage is None:
        return None
    prompt = int(_attr(usage, "prompt_tokens", 0) or 0)
    completion = int(_attr(usage, "completion_tokens", 0) or 0)

    # LiteLLM surfaces Anthropic cache counters on the normalized usage object.
    cache_write = int(_attr(usage, "cache_creation_input_tokens", 0) or 0)
    cache_read = int(_attr(usage, "cache_read_input_tokens", 0) or 0)
    if not cache_read:
        details = _attr(usage, "prompt_tokens_details")
        cache_read = int(_attr(details, "cached_tokens", 0) or 0) if details else 0
        # OpenAI-style: cached tokens are a SUBSET of prompt_tokens
        prompt = max(prompt - cache_read, 0)
    reasoning = 0
    cdetails = _attr(usage, "completion_tokens_details")
    if cdetails:
        reasoning = int(_attr(cdetails, "reasoning_tokens", 0) or 0)
        completion = max(completion - reasoning, 0)

    return TokenUsage(
        input_tokens=prompt,
        output_tokens=completion,
        cached_input_tokens=cache_read,
        cache_creation_tokens=cache_write,
        reasoning_tokens=reasoning,
    )


class StepCostLiteLLMLogger:
    """Plug into ``litellm.success_callback`` / ``failure_callback``."""

    def __init__(self, client: StepCost) -> None:
        self._client = client

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        model = str(kwargs.get("model") or "")
        active = self._client.start_span(
            kind=SpanKind.LLM_GENERATION,
            name=model,
            model=model,
            provider=_provider_from(kwargs, model),
        )
        usage = _usage_from_response(response_obj)
        if usage is not None:
            active._apply_usage(usage)
        span = active.finish()
        try:
            span.duration_ms = int((end_time - start_time).total_seconds() * 1000)
        except (TypeError, AttributeError):
            pass

    def log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        model = str(kwargs.get("model") or "")
        active = self._client.start_span(
            kind=SpanKind.LLM_GENERATION,
            name=model,
            model=model,
            provider=_provider_from(kwargs, model),
        )
        exc = kwargs.get("exception")
        active.set_metadata(error=type(exc).__name__ if exc is not None else "error")
        active.finish()
