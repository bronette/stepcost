"""Tests for provider usage extraction and invoice reconciliation."""

from decimal import Decimal

import pytest

from stepcost import cost_for_model, extract_openai_usage, naive_openai_usage
from stepcost.extractors.anthropic import extract_anthropic_usage
from stepcost.models import TokenUsage


def _invoice_openai(
    *,
    uncached_input: int,
    cached_input: int,
    visible_output: int,
    reasoning: int,
    model: str,
) -> Decimal:
    usage = TokenUsage(
        input_tokens=uncached_input,
        output_tokens=visible_output,
        cached_input_tokens=cached_input,
        reasoning_tokens=reasoning,
    )
    cost = cost_for_model(model, usage)
    assert cost is not None
    return cost.total_usd


def test_openai_cached_prompt_no_double_count():
    response = {
        "usage": {
            "prompt_tokens": 10_000,
            "completion_tokens": 2_000,
            "prompt_tokens_details": {"cached_tokens": 3_000},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    }
    usage = extract_openai_usage(response)
    assert usage.input_tokens == 7_000
    assert usage.cached_input_tokens == 3_000
    assert usage.output_tokens == 2_000

    correct = _invoice_openai(
        uncached_input=7_000,
        cached_input=3_000,
        visible_output=2_000,
        reasoning=0,
        model="gpt-4o-mini",
    )
    extracted_cost = cost_for_model("gpt-4o-mini", usage)
    assert extracted_cost is not None
    assert extracted_cost.total_usd == correct


def test_naive_openai_mapping_overcounts():
    response = {
        "usage": {
            "prompt_tokens": 10_000,
            "completion_tokens": 2_000,
            "prompt_tokens_details": {"cached_tokens": 3_000},
        }
    }
    naive = naive_openai_usage(response)
    correct = extract_openai_usage(response)
    naive_cost = cost_for_model("gpt-4o-mini", naive)
    correct_cost = cost_for_model("gpt-4o-mini", correct)
    assert naive_cost is not None and correct_cost is not None
    assert naive_cost.total_usd > correct_cost.total_usd
    # Cached input wrongly billed at full input rate on top of prompt_tokens.
    over_pct = (naive_cost.total_usd - correct_cost.total_usd) / correct_cost.total_usd
    assert over_pct > Decimal("0.10")


def test_openai_o_series_reasoning_split():
    response = {
        "usage": {
            "prompt_tokens": 5_000,
            "completion_tokens": 8_000,
            "prompt_tokens_details": {"cached_tokens": 1_000},
            "completion_tokens_details": {"reasoning_tokens": 6_000},
        }
    }
    usage = extract_openai_usage(response)
    assert usage.input_tokens == 4_000
    assert usage.cached_input_tokens == 1_000
    assert usage.output_tokens == 2_000
    assert usage.reasoning_tokens == 6_000

    correct = _invoice_openai(
        uncached_input=4_000,
        cached_input=1_000,
        visible_output=2_000,
        reasoning=6_000,
        model="o3-mini",
    )
    extracted_cost = cost_for_model("o3-mini", usage)
    assert extracted_cost is not None
    assert extracted_cost.total_usd == correct


def test_openai_reconciliation_against_hand_computed_invoice():
    """Extracted cost vs an invoice computed BY HAND from published rates.

    Deliberately not derived through cost_for_model — a wrong price, divisor,
    or extraction must fail this test (the previous version was circular).
    """
    # gpt-4o-mini: $0.15/1M input, $0.075/1M cached input, $0.60/1M output.
    # 50k prompt (20k cached) + 12k completion:
    #   30k * 0.15/1M = 0.0045; 20k * 0.075/1M = 0.0015; 12k * 0.60/1M = 0.0072
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 50_000,
                "completion_tokens": 12_000,
                "prompt_tokens_details": {"cached_tokens": 20_000},
            }
        }
    )
    cost = cost_for_model("gpt-4o-mini", usage)
    assert cost is not None
    assert cost.total_usd == Decimal("0.0132")

    # o3-mini: $1.10/1M input, $4.40/1M output+reasoning.
    # 8k uncached prompt, 4.5k completion of which 3k reasoning:
    #   8k * 1.10/1M = 0.0088; 1.5k * 4.40/1M = 0.0066; 3k * 4.40/1M = 0.0132
    usage = extract_openai_usage(
        {
            "usage": {
                "prompt_tokens": 8_000,
                "completion_tokens": 4_500,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 3_000},
            }
        }
    )
    cost = cost_for_model("o3-mini", usage)
    assert cost is not None
    assert cost.total_usd == Decimal("0.0286")


def test_anthropic_cache_lines():
    response = {
        "usage": {
            "input_tokens": 1_200,
            "output_tokens": 400,
            "cache_creation_input_tokens": 5_000,
            "cache_read_input_tokens": 8_000,
        }
    }
    usage = extract_anthropic_usage(response)
    assert usage.input_tokens == 1_200
    assert usage.output_tokens == 400
    assert usage.cache_creation_tokens == 5_000
    assert usage.cached_input_tokens == 8_000

    cost = cost_for_model("claude-sonnet-4-20250514", usage)
    assert cost is not None
    assert cost.total_usd > Decimal("0")


def test_anthropic_cache_rates_pinned_in_dollars():
    """Pin cache read = 0.1x input and 5m write = 1.25x / 1h write = 2x input.

    Hand-computed on claude-haiku-4-5 ($1/1M in, $5/1M out):
      100k in = 0.10; 20k out = 0.10; 200k cache-read * 0.10/1M = 0.02;
      50k 5m-write * 1.25/1M = 0.0625; 40k 1h-write * 2.00/1M = 0.08
    """
    usage = TokenUsage(
        input_tokens=100_000,
        output_tokens=20_000,
        cached_input_tokens=200_000,
        cache_creation_tokens=50_000,
        cache_creation_1h_tokens=40_000,
    )
    cost = cost_for_model("claude-haiku-4-5", usage)
    assert cost is not None
    assert cost.input_usd == Decimal("0.10")
    assert cost.output_usd == Decimal("0.10")
    assert cost.cached_input_usd == Decimal("0.02")
    assert cost.cache_creation_usd == Decimal("0.1425")  # 0.0625 + 0.08
    assert cost.total_usd == Decimal("0.3625")


def test_anthropic_1h_cache_ttl_split_extracted():
    response = {
        "usage": {
            "input_tokens": 1_000,
            "output_tokens": 100,
            "cache_creation_input_tokens": 5_000,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 2_000,
                "ephemeral_1h_input_tokens": 3_000,
            },
            "cache_read_input_tokens": 0,
        }
    }
    usage = extract_anthropic_usage(response)
    assert usage.cache_creation_tokens == 2_000
    assert usage.cache_creation_1h_tokens == 3_000

    # Collapsing the split to 1.25x would undercount the 1h line by 37.5%.
    cost = cost_for_model("claude-haiku-4-5", usage)
    assert cost is not None
    # 2k * 1.25/1M + 3k * 2.00/1M = 0.0025 + 0.006
    assert cost.cache_creation_usd == Decimal("0.0085")


def test_openai_responses_api_shape_extracts():
    """Responses API field names must not silently extract as zeros."""
    response = {
        "usage": {
            "input_tokens": 10_000,
            "output_tokens": 3_000,
            "input_tokens_details": {"cached_tokens": 4_000},
            "output_tokens_details": {"reasoning_tokens": 1_000},
        }
    }
    usage = extract_openai_usage(response)
    assert usage.input_tokens == 6_000
    assert usage.cached_input_tokens == 4_000
    assert usage.output_tokens == 2_000
    assert usage.reasoning_tokens == 1_000


def test_openai_embeddings_usage_extracts_as_embedding_tokens():
    response = {"usage": {"prompt_tokens": 12_345, "total_tokens": 12_345}}
    usage = extract_openai_usage(response)
    assert usage.embedding_tokens == 12_345
    assert usage.input_tokens == 0


def test_openai_unrecognized_usage_shape_raises():
    with pytest.raises(ValueError, match="Unrecognized OpenAI usage shape"):
        extract_openai_usage({"usage": {"tokens_used": 500}})


def test_openai_negative_tokens_raise():
    with pytest.raises(ValueError, match="negative"):
        extract_openai_usage(
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": -5},
                }
            }
        )


def test_openai_invalid_cached_raises():
    with pytest.raises(ValueError, match="cached_tokens"):
        extract_openai_usage({"usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                         "prompt_tokens_details": {"cached_tokens": 200}}})
