"""Tests for trace instrumentation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from stepcost import (
    StepCost,
    ContextRequiredError,
    Provider,
    SpanKind,
    TraceCostUnavailableError,
    agent_step,
    llm_call,
)


@pytest.fixture
def sqlite_sink(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'test.db'}"


def test_contextvar_agent_step_without_cc_threading(sqlite_sink: str):
    cc = StepCost(project="demo", sink=sqlite_sink, default_feature="copilot")

    with cc.trace(customer_id="acme", feature_id="support") as trace:
        with agent_step("plan") as step:
            step.set_metadata(tool="planner")
            with llm_call(model="gpt-4o-mini", provider="openai") as call:
                call.record_usage(input_tokens=1000, output_tokens=200)

    cc.flush()
    assert trace.total_usd > 0
    assert trace.by_kind["llm_generation"] > 0
    assert trace.by_step["plan"] > 0


def test_record_openai_response(sqlite_sink: str):
    cc = StepCost(project="demo", sink=sqlite_sink)
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10_000,
            completion_tokens=2_000,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3_000),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
    )

    with cc.trace() as trace:
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record(response)

    assert trace.total_usd > 0


def test_agent_step_outside_trace_raises():
    with pytest.raises(ContextRequiredError):
        with agent_step("orphan"):
            pass


def test_trace_total_usd_works_with_stdout_sink():
    cc = StepCost(project="demo", sink="stdout")

    with cc.trace() as trace:
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record_usage(input_tokens=1_000_000, output_tokens=0)

    assert trace.total_usd > 0


def test_unknown_trace_raises():
    cc = StepCost(project="demo", sink="stdout")
    with pytest.raises(TraceCostUnavailableError):
        cc.trace_total_usd("nonexistent-trace-id")


def test_span_kinds(sqlite_sink: str):
    cc = StepCost(project="demo", sink=sqlite_sink)

    with cc.trace() as trace:
        with cc.span(kind=SpanKind.RETRIEVAL, name="pinecone") as retrieval:
            retrieval.record_usage(input_tokens=500)

    cc.flush()
    assert trace.total_usd == 0.0


def test_embedding_span_rolls_up(sqlite_sink: str):
    cc = StepCost(project="demo", sink=sqlite_sink)

    with cc.trace(feature_id="rag") as trace:
        with agent_step("retrieve"):
            with cc.span(
                kind=SpanKind.EMBEDDING,
                name="text-embedding-3-small",
                model="text-embedding-3-small",
                provider=Provider.OPENAI,
            ) as emb:
                emb.set_metadata(tool="pinecone")
                emb.record_usage(embedding_tokens=1_200_000)

    cc.flush()
    assert trace.total_usd > 0
    assert trace.by_kind["embedding"] > 0
    assert trace.by_step["retrieve"] > 0
