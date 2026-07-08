"""Tests for LiteLLM callback + OpenAI/Anthropic client wrappers (no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from stepcost import StepCost, agent_step
from stepcost.integrations.anthropic import instrument_anthropic
from stepcost.integrations.litellm import StepCostLiteLLMLogger
from stepcost.integrations.openai import instrument_openai
from stepcost.models import SpanKind


@pytest.fixture
def cc() -> StepCost:
    return StepCost(project="wrap-demo", sink="stdout")


# --------------------------------------------------------------------------- #
# LiteLLM callback
# --------------------------------------------------------------------------- #
def test_litellm_success_openai_shape(cc: StepCost):
    logger = StepCostLiteLLMLogger(cc)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10_000,
            completion_tokens=500,
            prompt_tokens_details=SimpleNamespace(cached_tokens=4_000),
            completion_tokens_details=None,
        )
    )
    t0 = datetime.now(timezone.utc)
    with cc.trace() as trace:
        logger.log_success_event(
            {"model": "gpt-4o-mini", "litellm_params": {"custom_llm_provider": "openai"}},
            response, t0, t0 + timedelta(milliseconds=340),
        )
    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.usage.input_tokens == 6_000  # cached subtracted, no double count
    assert span.usage.cached_input_tokens == 4_000
    assert span.duration_ms == 340
    assert trace.total_usd > 0


def test_litellm_anthropic_cache_counters(cc: StepCost):
    logger = StepCostLiteLLMLogger(cc)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=1_000,
            completion_tokens=100,
            cache_creation_input_tokens=8_000,
            cache_read_input_tokens=2_000,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        )
    )
    with cc.trace():
        logger.log_success_event({"model": "claude-haiku-4-5"}, response, None, None)
    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.provider.value == "anthropic"
    assert span.usage.cache_creation_tokens == 8_000
    assert span.usage.cached_input_tokens == 2_000
    # 1k*1.00 + 100*5.00 + 8k*1.25 + 2k*0.10 per 1M = 0.001+0.0005+0.01+0.0002
    assert span.cost.total_usd == Decimal("0.0117")


def test_litellm_failure_records_error_span(cc: StepCost):
    logger = StepCostLiteLLMLogger(cc)
    with cc.trace():
        logger.log_failure_event(
            {"model": "gpt-4o-mini", "exception": TimeoutError("slow")}, None, None, None
        )
    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.tags["error"] == "TimeoutError"


# --------------------------------------------------------------------------- #
# OpenAI wrapper
# --------------------------------------------------------------------------- #
def _fake_openai_client():
    def chat_create(**kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=2_000, completion_tokens=300,
                prompt_tokens_details=None, completion_tokens_details=None,
            )
        )

    def emb_create(**kwargs):
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=5_000, total_tokens=5_000))

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat_create)),
        responses=None,
        embeddings=SimpleNamespace(create=emb_create),
    )


def test_openai_wrapper_prices_calls_and_nests_under_steps(cc: StepCost):
    client = instrument_openai(_fake_openai_client(), cc)
    with cc.trace() as trace:
        with agent_step("answer") as step:
            client.chat.completions.create(model="gpt-4o-mini", messages=[])
            client.embeddings.create(model="text-embedding-3-small", input="x")
    llm = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    emb = next(s for s in cc._span_index.values() if s.kind == SpanKind.EMBEDDING)
    assert llm.parent_span_id == step.span.span_id
    assert emb.parent_span_id == step.span.span_id
    assert emb.usage.embedding_tokens == 5_000
    assert trace.by_step["answer"] > 0


def test_openai_wrapper_closes_span_on_provider_error(cc: StepCost):
    client = _fake_openai_client()

    def boom(**kwargs):
        raise ConnectionError("down")

    client.chat.completions.create = boom
    instrument_openai(client, cc)
    with cc.trace():
        with pytest.raises(ConnectionError):
            client.chat.completions.create(model="gpt-4o-mini")
    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.tags["error"] == "ConnectionError"


# --------------------------------------------------------------------------- #
# Anthropic wrapper
# --------------------------------------------------------------------------- #
def test_anthropic_wrapper_extracts_cache_ttl_split(cc: StepCost):
    def create(**kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1_000,
                output_tokens=50,
                cache_creation_input_tokens=5_000,
                cache_read_input_tokens=0,
                cache_creation=SimpleNamespace(
                    ephemeral_5m_input_tokens=2_000, ephemeral_1h_input_tokens=3_000
                ),
            )
        )

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    instrument_anthropic(client, cc)
    with cc.trace() as trace:
        client.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=[])
    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.usage.cache_creation_tokens == 2_000
    assert span.usage.cache_creation_1h_tokens == 3_000
    assert trace.total_usd > 0
