"""Tests for cost computation."""

from decimal import Decimal

from stepcost.models import ModelPricing, Provider, TokenUsage
from stepcost.pricing import compute_cost, cost_for_model, default_price_table


def test_gpt4o_mini_cost():
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=500_000)
    cost = cost_for_model("gpt-4o-mini", usage)
    assert cost is not None
    assert cost.input_usd == Decimal("0.15")
    assert cost.output_usd == Decimal("0.30")
    assert cost.total_usd == Decimal("0.45")
    assert cost.price_table_version == default_price_table().version


def test_cached_input_pricing():
    usage = TokenUsage(cached_input_tokens=1_000_000)
    pricing = ModelPricing(
        provider=Provider.OPENAI,
        input_per_1m=Decimal("2.50"),
        output_per_1m=Decimal("10.00"),
        cached_input_per_1m=Decimal("1.25"),
    )
    cost = compute_cost(usage, pricing, price_table_version="test")
    assert cost.cached_input_usd == Decimal("1.25")
    assert cost.total_usd == Decimal("1.25")


def test_embedding_pricing():
    usage = TokenUsage(embedding_tokens=1_000_000)
    pricing = ModelPricing(
        provider=Provider.OPENAI,
        input_per_1m=Decimal("0"),
        output_per_1m=Decimal("0"),
        embedding_per_1m=Decimal("0.02"),
    )
    cost = compute_cost(usage, pricing, price_table_version="test")
    assert cost.embedding_usd == Decimal("0.02")
    assert cost.total_usd == Decimal("0.02")


def test_embedding_model_from_price_table():
    usage = TokenUsage(embedding_tokens=1_000_000)
    cost = cost_for_model("text-embedding-3-small", usage)
    assert cost is not None
    assert cost.embedding_usd == Decimal("0.02")
    assert cost.total_usd == Decimal("0.02")


def test_unknown_model_returns_none():
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    assert cost_for_model("not-a-real-model", usage) is None
