"""Tests for the LangChain callback handler (no network, synthetic callbacks)."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from stepcost import StepCost
from stepcost.integrations.langchain import StepCostCallbackHandler
from stepcost.models import SpanKind


@pytest.fixture
def cc() -> StepCost:
    return StepCost(project="lc-demo", sink="stdout")


def _llm_result_with_usage_metadata(**meta) -> LLMResult:
    msg = AIMessage(content="hi", usage_metadata=meta)
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


def test_agent_run_tree_maps_to_priced_spans(cc: StepCost):
    handler = StepCostCallbackHandler(cc)
    chain_id, llm_id, tool_id = uuid4(), uuid4(), uuid4()

    with cc.trace(feature_id="support") as trace:
        handler.on_chain_start({"name": "answer"}, {}, run_id=chain_id)
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
            [],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model": "gpt-4o-mini", "_type": "openai-chat"},
        )
        handler.on_llm_end(
            _llm_result_with_usage_metadata(
                input_tokens=1_000, output_tokens=200, total_tokens=1_200
            ),
            run_id=llm_id,
        )
        handler.on_tool_start({"name": "search"}, "query", run_id=tool_id, parent_run_id=chain_id)
        handler.on_tool_end("result", run_id=tool_id)
        handler.on_chain_end({}, run_id=chain_id)

    assert trace.total_usd > 0
    assert trace.by_step["answer"] > 0  # llm cost attributed to the chain step
    # Verify the parent chain: llm + tool under the "answer" step
    spans = list(cc._span_index.values())
    step = next(s for s in spans if s.kind == SpanKind.AGENT_STEP and s.name == "answer")
    llm = next(s for s in spans if s.kind == SpanKind.LLM_GENERATION)
    tool = next(s for s in spans if s.kind == SpanKind.TOOL_CALL)
    assert llm.parent_span_id == step.span_id
    assert tool.parent_span_id == step.span_id
    assert step.parent_span_id == trace.root_span.span_id


def test_usage_metadata_cache_details_not_double_counted(cc: StepCost):
    """LangChain folds cache reads/writes INTO input_tokens — we must unfold."""
    handler = StepCostCallbackHandler(cc)
    llm_id = uuid4()

    with cc.trace():
        handler.on_chat_model_start(
            {"id": ["ChatAnthropic"]},
            [],
            run_id=llm_id,
            invocation_params={"model": "claude-haiku-4-5", "_type": "anthropic-chat"},
        )
        handler.on_llm_end(
            _llm_result_with_usage_metadata(
                input_tokens=10_000,  # includes the cache figures below
                output_tokens=500,
                total_tokens=10_500,
                input_token_details={"cache_read": 6_000, "cache_creation": 3_000},
            ),
            run_id=llm_id,
        )

    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.usage.input_tokens == 1_000  # 10k - 6k read - 3k write
    assert span.usage.cached_input_tokens == 6_000
    assert span.usage.cache_creation_tokens == 3_000
    # Hand-computed on haiku-4-5 ($/1M): 1k*1.00 + 6k*0.10 + 3k*1.25 + 0.5k*5.00
    #   = 0.001 + 0.0006 + 0.00375 + 0.0025
    assert span.cost.total_usd == Decimal("0.007850")


def test_passthrough_runnables_create_no_spans_but_bridge_parents(cc: StepCost):
    handler = StepCostCallbackHandler(cc)
    step_id, seq_id, llm_id = uuid4(), uuid4(), uuid4()

    with cc.trace():
        handler.on_chain_start({"name": "plan"}, {}, run_id=step_id)
        # LCEL structural node between the step and the llm
        handler.on_chain_start({"name": "RunnableSequence"}, {}, run_id=seq_id, parent_run_id=step_id)
        handler.on_chat_model_start(
            {"id": ["ChatOpenAI"]},
            [],
            run_id=llm_id,
            parent_run_id=seq_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_end(
            _llm_result_with_usage_metadata(input_tokens=100, output_tokens=10, total_tokens=110),
            run_id=llm_id,
        )
        handler.on_chain_end({}, run_id=seq_id)
        handler.on_chain_end({}, run_id=step_id)

    spans = list(cc._span_index.values())
    assert not any(s.name == "RunnableSequence" for s in spans)
    step = next(s for s in spans if s.name == "plan")
    llm = next(s for s in spans if s.kind == SpanKind.LLM_GENERATION)
    assert llm.parent_span_id == step.span_id  # bridged through the sequence


def test_openai_token_usage_fallback(cc: StepCost):
    handler = StepCostCallbackHandler(cc)
    llm_id = uuid4()
    result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="x"))]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 5_000,
                "completion_tokens": 300,
                "prompt_tokens_details": {"cached_tokens": 2_000},
            }
        },
    )
    with cc.trace():
        handler.on_llm_start(
            {"id": ["OpenAI"]}, ["p"], run_id=llm_id, invocation_params={"model": "gpt-4o-mini"}
        )
        handler.on_llm_end(result, run_id=llm_id)

    span = next(s for s in cc._span_index.values() if s.kind == SpanKind.LLM_GENERATION)
    assert span.usage.input_tokens == 3_000
    assert span.usage.cached_input_tokens == 2_000


def test_error_paths_still_close_spans(cc: StepCost):
    handler = StepCostCallbackHandler(cc)
    chain_id, llm_id = uuid4(), uuid4()
    with cc.trace():
        handler.on_chain_start({"name": "step"}, {}, run_id=chain_id)
        handler.on_llm_start(
            {"id": ["ChatOpenAI"]}, ["p"], run_id=llm_id, parent_run_id=chain_id,
            invocation_params={"model": "gpt-4o-mini"},
        )
        handler.on_llm_error(RuntimeError("rate limited"), run_id=llm_id)
        handler.on_chain_error(RuntimeError("boom"), run_id=chain_id)

    spans = [s for s in cc._span_index.values() if s.kind != SpanKind.TRACE]
    assert len(spans) == 2
    llm = next(s for s in spans if s.kind == SpanKind.LLM_GENERATION)
    assert llm.tags.get("error") == "RuntimeError"
    assert handler._runs == {}  # nothing leaked
