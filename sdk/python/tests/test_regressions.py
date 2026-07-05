"""Regression tests for data-loss, concurrency, pricing-immutability, waste,
and CLI bugs found in the 2026-07-05 pre-MVP review."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from decimal import Decimal
from pathlib import Path

import pytest

from stepcost import StepCost, Provider, SpanKind, agent_step, llm_call
from stepcost.cli import main as cli_main
from stepcost.models import ModelPricing, Span, TokenUsage
from stepcost.pricing import compute_cost, default_price_table, register_custom_model
from stepcost.report import build_summary, unpriced_models
from stepcost.waste import detect_waste


def _db_span_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Data loss
# --------------------------------------------------------------------------- #
def test_spans_persist_at_trace_exit_without_explicit_flush(tmp_path: Path):
    db = tmp_path / "t.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with cc.trace():
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record_usage(input_tokens=100, output_tokens=10)
    # No cc.flush() — the trace boundary must persist on its own.
    assert _db_span_count(db) == 2  # llm span + trace root


def test_span_persisted_when_body_raises(tmp_path: Path):
    db = tmp_path / "t.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with pytest.raises(RuntimeError):
        with cc.trace():
            with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
                call.record_usage(input_tokens=100, output_tokens=10)
                raise RuntimeError("provider exploded")
    assert _db_span_count(db) == 2


class _FlakySink:
    def __init__(self) -> None:
        self.healthy = False
        self.emitted: list = []

    def emit(self, spans) -> None:
        if not self.healthy:
            raise OSError("disk full")
        self.emitted.extend(spans)

    def flush(self) -> None:
        return None


def test_sink_failure_requeues_batch_instead_of_dropping():
    sink = _FlakySink()
    cc = StepCost(project="demo", sink=sink)
    with cc.trace():
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record_usage(input_tokens=100, output_tokens=10)
    # Trace-exit flush failed silently (sink down) — spans must still be pending.
    assert len(cc._pending) == 2
    with pytest.raises(OSError):
        cc.flush()
    assert len(cc._pending) == 2  # explicit flush raises, but nothing was lost
    sink.healthy = True
    cc.flush()
    assert len(sink.emitted) == 2
    assert cc._pending == []


def test_close_flushes_and_releases_sink(tmp_path: Path):
    db = tmp_path / "t.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with cc.trace():
        with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
            call.record_usage(input_tokens=5)
    cc.close()
    assert _db_span_count(db) == 2


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_concurrent_async_tasks_parent_to_their_own_step():
    cc = StepCost(project="demo", sink="stdout")

    async def worker(name: str) -> tuple[str, str]:
        with agent_step(name) as step:
            await asyncio.sleep(0.01)
            with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
                await asyncio.sleep(0.01)
                call.record_usage(input_tokens=10)
            return step.span.span_id, call.span.span_id

    async def run() -> list[tuple[str, str]]:
        return await asyncio.gather(worker("a"), worker("b"))

    with cc.trace() as trace:
        pairs = asyncio.run(run())

    for step_id, call_id in pairs:
        call_span = cc._span_index[call_id]
        step_span = cc._span_index[step_id]
        # Each llm call must parent to ITS OWN step, not whichever step another
        # task happened to have open (the old shared-list stack corrupted this).
        assert call_span.parent_span_id == step_id
        assert step_span.parent_span_id == trace.root_span.span_id


def test_flush_from_worker_thread_is_safe(tmp_path: Path):
    db = tmp_path / "t.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with cc.trace():
        for _ in range(5):
            with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
                call.record_usage(input_tokens=10)

    errors: list[BaseException] = []

    def worker() -> None:
        try:
            cc.flush()
        except BaseException as exc:  # noqa: BLE001 — test must capture anything
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors == []
    assert _db_span_count(db) == 6


# --------------------------------------------------------------------------- #
# Pricing safety
# --------------------------------------------------------------------------- #
def test_model_pricing_is_frozen():
    pricing = default_price_table().models["gpt-4o"]
    with pytest.raises(Exception):
        pricing.input_per_1m = Decimal("0.01")  # type: ignore[misc]


def test_register_custom_model_does_not_touch_default_table():
    table = register_custom_model(
        "my-model",
        ModelPricing(provider=Provider.OTHER, input_per_1m=Decimal("1"), output_per_1m=Decimal("2")),
    )
    assert "my-model" in table.models
    assert "my-model" not in default_price_table().models


def test_zero_reasoning_rate_means_free_not_output_rate():
    pricing = ModelPricing(
        provider=Provider.OTHER,
        input_per_1m=Decimal("1"),
        output_per_1m=Decimal("5"),
        reasoning_per_1m=Decimal("0"),
    )
    cost = compute_cost(
        TokenUsage(reasoning_tokens=1_000_000), pricing, price_table_version="test"
    )
    assert cost.reasoning_usd == Decimal("0")


def test_unpriced_models_surface_in_report():
    span = Span(
        trace_id="t1",
        kind=SpanKind.LLM_GENERATION,
        project_id="demo",
        model="some-brand-new-model",
        usage=TokenUsage(input_tokens=1_000, output_tokens=100),
    )
    assert unpriced_models([span]) == {"some-brand-new-model": 1}
    summary = build_summary([span])
    assert summary.unpriced == {"some-brand-new-model": 1}


# --------------------------------------------------------------------------- #
# Waste heuristics
# --------------------------------------------------------------------------- #
def _llm_span(trace_id: str, *, input_tokens: int, cached: int = 0, cache_write: int = 0) -> Span:
    return Span(
        trace_id=trace_id,
        kind=SpanKind.LLM_GENERATION,
        project_id="demo",
        model="claude-haiku-4-5",
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=300,
            cached_input_tokens=cached,
            cache_creation_tokens=cache_write,
        ),
    )


def test_missing_cache_not_flagged_when_traffic_writes_cache():
    spans = [_llm_span("t1", input_tokens=8_000, cache_write=8_000) for _ in range(4)]
    codes = {w.code for w in detect_waste(spans)}
    assert "missing_cache" not in codes


def test_missing_cache_still_flagged_for_truly_uncached_traffic():
    spans = [_llm_span("t1", input_tokens=8_000) for _ in range(4)]
    codes = {w.code for w in detect_waste(spans)}
    assert "missing_cache" in codes


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
@pytest.fixture
def populated_db(tmp_path: Path) -> tuple[str, str]:
    db = tmp_path / "cli.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    with cc.trace(feature_id="support") as trace:
        with agent_step("plan"):
            with llm_call(model="gpt-4o-mini", provider=Provider.OPENAI) as call:
                call.record_usage(input_tokens=1_000, output_tokens=200)
    cc.flush()
    return str(db), trace.trace_id


def test_cli_summary_json(populated_db, capsys):
    db, _tid = populated_db
    assert cli_main(["report", db, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_traces"] == 1
    assert Decimal(payload["total_usd"]) > 0


def test_cli_trace_json_includes_tree(populated_db, capsys):
    db, tid = populated_db
    assert cli_main(["report", db, "--trace", tid, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["trace_id"] == tid
    assert payload["tree"], "trace --json must include the span tree"
    root = payload["tree"][0]
    assert root["kind"] == "trace"
    step = root["children"][0]
    assert step["kind"] == "agent_step"
    assert step["children"][0]["kind"] == "llm_generation"


def test_cli_sqlite_url_accepted(populated_db, capsys):
    db, _tid = populated_db
    assert cli_main(["report", f"sqlite:///{db}", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_spans"] == 3


def test_cli_non_stepcost_db_clean_error(tmp_path: Path, capsys):
    bogus = tmp_path / "bogus.db"
    bogus.write_bytes(b"definitely not sqlite")
    assert cli_main(["report", str(bogus)]) == 2
    assert "error:" in capsys.readouterr().err
