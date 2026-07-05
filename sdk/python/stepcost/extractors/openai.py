"""OpenAI usage extraction — Chat Completions, Responses API, embeddings.

OpenAI billing semantics (do NOT double-count):
- Chat: ``prompt_tokens`` is the full prompt size and *includes* cached tokens
  (``prompt_tokens_details.cached_tokens``); ``completion_tokens`` includes
  reasoning tokens (``completion_tokens_details.reasoning_tokens``).
- Responses API: same containment rules, different names — ``input_tokens`` /
  ``input_tokens_details.cached_tokens`` and ``output_tokens`` /
  ``output_tokens_details.reasoning_tokens``.
- Embeddings: usage carries only ``prompt_tokens`` + ``total_tokens``; billed
  at the embedding rate.
- Uncached input billed at input rate; cached subset at the cached rate.
- Visible output billed at output rate; reasoning at the reasoning rate
  (falls back to output rate in pricing).

An unrecognized usage shape raises instead of extracting zeros — silently
recording $0.00 is the worst failure mode for a cost-accuracy SDK.
"""

from __future__ import annotations

from typing import Any

from stepcost.models import TokenUsage


def _raw(obj: Any, *path: str) -> Any:
    current = obj
    for key in path:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
    return current


def _get(obj: Any, *path: str, default: int = 0) -> int:
    value = _raw(obj, *path)
    return default if value is None else int(value)


def _check_subsets(
    input_total: int, cached: int, output_total: int, reasoning: int
) -> None:
    for label, value in (
        ("input", input_total),
        ("cached", cached),
        ("output", output_total),
        ("reasoning", reasoning),
    ):
        if value < 0:
            raise ValueError(f"OpenAI usage has negative {label} tokens ({value})")
    if cached > input_total:
        raise ValueError(f"cached_tokens ({cached}) exceeds input tokens ({input_total})")
    if reasoning > output_total:
        raise ValueError(
            f"reasoning_tokens ({reasoning}) exceeds output tokens ({output_total})"
        )


def extract_openai_usage(response: Any) -> TokenUsage:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        raise ValueError("OpenAI response missing usage")

    prompt_tokens = _raw(usage, "prompt_tokens")
    completion_tokens = _raw(usage, "completion_tokens")
    input_tokens = _raw(usage, "input_tokens")
    output_tokens = _raw(usage, "output_tokens")

    if prompt_tokens is not None and completion_tokens is None and input_tokens is None:
        # Embeddings-style usage: prompt_tokens + total_tokens only.
        tokens = int(prompt_tokens)
        if tokens < 0:
            raise ValueError(f"OpenAI usage has negative prompt_tokens ({tokens})")
        return TokenUsage(embedding_tokens=tokens)

    if prompt_tokens is not None or completion_tokens is not None:
        # Chat Completions.
        in_total = int(prompt_tokens or 0)
        out_total = int(completion_tokens or 0)
        cached = _get(usage, "prompt_tokens_details", "cached_tokens")
        reasoning = _get(usage, "completion_tokens_details", "reasoning_tokens")
    elif input_tokens is not None or output_tokens is not None:
        # Responses API.
        in_total = int(input_tokens or 0)
        out_total = int(output_tokens or 0)
        cached = _get(usage, "input_tokens_details", "cached_tokens")
        reasoning = _get(usage, "output_tokens_details", "reasoning_tokens")
    else:
        raise ValueError(
            "Unrecognized OpenAI usage shape: expected Chat Completions "
            "(prompt_tokens/completion_tokens), Responses API "
            "(input_tokens/output_tokens), or embeddings (prompt_tokens) fields; "
            f"got {usage!r}"
        )

    _check_subsets(in_total, cached, out_total, reasoning)

    return TokenUsage(
        input_tokens=in_total - cached,
        output_tokens=out_total - reasoning,
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )


def naive_openai_usage(response: Any) -> TokenUsage:
    """WRONG mapping — prompt_tokens + cached_tokens double-counts. For tests only."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        raise ValueError("OpenAI response missing usage")
    prompt_tokens = _get(usage, "prompt_tokens")
    completion_tokens = _get(usage, "completion_tokens")
    cached_tokens = _get(usage, "prompt_tokens_details", "cached_tokens")
    reasoning_tokens = _get(usage, "completion_tokens_details", "reasoning_tokens")
    return TokenUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        cached_input_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
