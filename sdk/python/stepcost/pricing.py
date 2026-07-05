"""Versioned price table and cost computation."""

from __future__ import annotations

import json
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from pathlib import Path

from stepcost.models import CostBreakdown, ModelPricing, PriceTable, TokenUsage

_PER_M = Decimal("1000000")


def _per_million(tokens: int, rate_per_1m: Decimal) -> Decimal:
    if tokens <= 0 or rate_per_1m <= 0:
        return Decimal("0")
    return (Decimal(tokens) / _PER_M) * rate_per_1m


def compute_cost(
    usage: TokenUsage,
    pricing: ModelPricing,
    *,
    price_table_version: str,
) -> CostBreakdown:
    input_usd = _per_million(usage.input_tokens, pricing.input_per_1m)
    output_usd = _per_million(usage.output_tokens, pricing.output_per_1m)

    cached_rate = pricing.cached_input_per_1m or Decimal("0")
    cached_input_usd = _per_million(usage.cached_input_tokens, cached_rate)

    # `is not None`, not truthiness: an explicit 0 reasoning rate must mean
    # free, not "fall back to the output rate".
    reasoning_rate = (
        pricing.reasoning_per_1m if pricing.reasoning_per_1m is not None else pricing.output_per_1m
    )
    reasoning_usd = _per_million(usage.reasoning_tokens, reasoning_rate)

    cache_write_rate = pricing.cache_write_per_1m or Decimal("0")
    cache_write_1h_rate = (
        pricing.cache_write_1h_per_1m
        if pricing.cache_write_1h_per_1m is not None
        else cache_write_rate
    )
    cache_creation_usd = _per_million(
        usage.cache_creation_tokens, cache_write_rate
    ) + _per_million(usage.cache_creation_1h_tokens, cache_write_1h_rate)

    embedding_rate = pricing.embedding_per_1m or Decimal("0")
    embedding_usd = _per_million(usage.embedding_tokens, embedding_rate)

    total = (
        input_usd + output_usd + cached_input_usd
        + reasoning_usd + cache_creation_usd + embedding_usd
    )
    return CostBreakdown(
        input_usd=input_usd,
        output_usd=output_usd,
        cached_input_usd=cached_input_usd,
        reasoning_usd=reasoning_usd,
        cache_creation_usd=cache_creation_usd,
        embedding_usd=embedding_usd,
        total_usd=total,
        price_table_version=price_table_version,
    )


def load_price_table(path: Path | None = None) -> PriceTable:
    if path is not None:
        data = json.loads(path.read_text())
        return PriceTable.model_validate(data)

    raw = resources.files("stepcost.data").joinpath("price_table.json").read_text()
    return PriceTable.model_validate(json.loads(raw))


@lru_cache(maxsize=1)
def default_price_table() -> PriceTable:
    return load_price_table()


def lookup_model(model: str, table: PriceTable | None = None) -> ModelPricing | None:
    table = table or default_price_table()
    return table.models.get(model)


def cost_for_model(
    model: str,
    usage: TokenUsage,
    *,
    table: PriceTable | None = None,
) -> CostBreakdown | None:
    table = table or default_price_table()
    pricing = lookup_model(model, table)
    if pricing is None:
        return None
    return compute_cost(usage, pricing, price_table_version=table.version)


def register_custom_model(
    model: str,
    pricing: ModelPricing,
    *,
    table: PriceTable | None = None,
) -> PriceTable:
    """Return a new table with an additional model (immutable pattern)."""
    table = table or default_price_table()
    models = dict(table.models)
    models[model] = pricing
    return PriceTable(version=table.version, models=models)
