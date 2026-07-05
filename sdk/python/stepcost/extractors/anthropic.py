"""Anthropic Messages API usage extraction.

Anthropic reports separate counters — no overlap:
- input_tokens, output_tokens
- cache_creation_input_tokens, cache_read_input_tokens
- usage.cache_creation.{ephemeral_5m_input_tokens, ephemeral_1h_input_tokens}
  splits the write total by TTL; 1h writes bill at 2x input vs 1.25x for 5m,
  so the split must be preserved for invoice-grade cost.
"""

from __future__ import annotations

from typing import Any

from stepcost.models import TokenUsage


def _get(obj: Any, *path: str, default: int = 0) -> int:
    current = obj
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    if current is None:
        return default
    return int(current)


def extract_anthropic_usage(response: Any) -> TokenUsage:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        raise ValueError("Anthropic response missing usage")

    cache_write_5m = _get(usage, "cache_creation", "ephemeral_5m_input_tokens")
    cache_write_1h = _get(usage, "cache_creation", "ephemeral_1h_input_tokens")
    if cache_write_5m == 0 and cache_write_1h == 0:
        # No TTL breakdown in the response — treat the aggregate as 5m-TTL
        # writes (the API default when no ttl is requested).
        cache_write_5m = _get(usage, "cache_creation_input_tokens")

    return TokenUsage(
        input_tokens=_get(usage, "input_tokens"),
        output_tokens=_get(usage, "output_tokens"),
        cache_creation_tokens=cache_write_5m,
        cache_creation_1h_tokens=cache_write_1h,
        cached_input_tokens=_get(usage, "cache_read_input_tokens"),
    )
