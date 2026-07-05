"""Provider response → TokenUsage extraction (invoice-safe, no double-count)."""

from __future__ import annotations

from typing import Any

from stepcost.extractors.anthropic import extract_anthropic_usage
from stepcost.extractors.openai import extract_openai_usage
from stepcost.models import Provider, TokenUsage


def extract_usage(response: Any, *, provider: Provider) -> TokenUsage:
    if provider == Provider.OPENAI:
        return extract_openai_usage(response)
    if provider == Provider.ANTHROPIC:
        return extract_anthropic_usage(response)
    raise ValueError(f"No usage extractor for provider {provider!r}")
