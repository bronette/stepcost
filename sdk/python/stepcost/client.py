"""StepCost client — trace instrumentation entry point."""

from __future__ import annotations

import atexit
import logging
import threading
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterator
from uuid import uuid4

from stepcost.context import ActiveSpan, TraceContext
from stepcost.errors import TraceCostUnavailableError
from stepcost.models import PriceTable, Provider, Span, SpanKind
from stepcost.pricing import default_price_table
from stepcost.rollups import TraceRollup
from stepcost.runtime import current_span, get_trace
from stepcost.sinks.base import SpanSink
from stepcost.sinks.sqlite import SQLiteSink
from stepcost.sinks.stdout import StdoutSink

logger = logging.getLogger("stepcost")

# Weak so clients can be garbage-collected; strong atexit registration per
# instance would pin every client (and its sink) for the process lifetime.
_LIVE_CLIENTS: weakref.WeakSet[StepCost] = weakref.WeakSet()


def _flush_all_at_exit() -> None:
    for client in list(_LIVE_CLIENTS):
        client._safe_flush()


atexit.register(_flush_all_at_exit)


def _resolve_sink(sink: str | SpanSink) -> SpanSink:
    if not isinstance(sink, str):
        return sink
    if sink == "stdout":
        return StdoutSink()
    if sink.startswith("sqlite://"):
        return SQLiteSink.from_url(sink)
    raise ValueError(f"Unknown sink {sink!r}. Use 'stdout' or 'sqlite:///path'")


class StepCost:
    def __init__(
        self,
        *,
        project: str,
        environment: str = "dev",
        sink: str | SpanSink = "stdout",
        default_feature: str | None = None,
        default_customer_id: str | None = None,
        default_team_id: str | None = None,
        price_table: PriceTable | None = None,
    ) -> None:
        self.project = project
        self.environment = environment
        self.default_feature = default_feature
        self.default_customer_id = default_customer_id
        self.default_team_id = default_team_id
        self.price_table = price_table or default_price_table()
        self._sink = _resolve_sink(sink)
        self._pending: list[Span] = []
        self._batch_size = 50
        # RLock: flush() holds it across the sink write so a concurrent flush
        # can't double-emit; emitters block briefly during a batched write.
        self._lock = threading.RLock()
        # Insertion-ordered so old traces can be evicted; unbounded growth here
        # was a leak in long-running servers.
        self._trace_rollups: OrderedDict[str, TraceRollup] = OrderedDict()
        self._trace_spans: OrderedDict[str, list[str]] = OrderedDict()
        self._span_index: dict[str, Span] = {}
        self._max_traces = 1024
        _LIVE_CLIENTS.add(self)

    def trace(
        self,
        *,
        customer_id: str | None = None,
        feature_id: str | None = None,
        team_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        name: str | None = None,
    ) -> TraceContext:
        return TraceContext(
            self,
            customer_id=customer_id,
            feature_id=feature_id,
            team_id=team_id,
            session_id=session_id,
            user_id=user_id,
            name=name,
        )

    @contextmanager
    def span(
        self,
        *,
        kind: SpanKind,
        name: str | None = None,
        model: str | None = None,
        provider: Provider | None = None,
        prompt_version: str | None = None,
    ) -> Iterator[ActiveSpan]:
        trace = get_trace()
        trace_id = trace.trace_id if trace else str(uuid4())
        enclosing = current_span()
        if enclosing is not None:
            parent_id = enclosing.span.span_id
        elif trace and trace.root_span:
            parent_id = trace.root_span.span_id
        else:
            parent_id = None

        span = Span(
            trace_id=trace_id,
            parent_span_id=parent_id if kind != SpanKind.TRACE else None,
            kind=kind,
            name=name,
            project_id=self.project,
            environment=self.environment,
            customer_id=trace.customer_id if trace else self.default_customer_id,
            feature_id=trace.feature_id if trace else self.default_feature,
            team_id=trace.team_id if trace else self.default_team_id,
            session_id=trace.session_id if trace else None,
            user_id=trace.user_id if trace else None,
            model=model,
            provider=provider,
            prompt_version=prompt_version,
        )
        self._index_span(span)
        active = ActiveSpan(client=self, span=span)
        active.__enter__()
        try:
            yield active
        finally:
            active.finish()

    def _init_trace_rollup(self, trace_id: str) -> None:
        with self._lock:
            self._trace_rollups[trace_id] = TraceRollup()

    def _index_span(self, span: Span) -> None:
        with self._lock:
            self._span_index[span.span_id] = span
            if span.trace_id not in self._trace_spans:
                self._trace_spans[span.trace_id] = []
                self._evict_old_traces()
            self._trace_spans[span.trace_id].append(span.span_id)

    def _evict_old_traces(self) -> None:
        while len(self._trace_spans) > self._max_traces:
            trace_id, span_ids = self._trace_spans.popitem(last=False)
            self._trace_rollups.pop(trace_id, None)
            for span_id in span_ids:
                self._span_index.pop(span_id, None)

    def _resolve_agent_step(self, span: Span) -> str | None:
        # Start at the span itself: cost recorded directly on an agent_step
        # belongs to that step, not its enclosing one.
        current: Span | None = span
        while current is not None:
            if current.kind == SpanKind.AGENT_STEP:
                return current.name
            parent_id = current.parent_span_id
            current = self._span_index.get(parent_id) if parent_id else None
        return None

    def _emit(self, span: Span) -> None:
        with self._lock:
            self._span_index[span.span_id] = span
            rollup = self._trace_rollups.get(span.trace_id)
            if rollup is not None:
                rollup.add(span, agent_step=self._resolve_agent_step(span))
            self._pending.append(span)
            batch_full = len(self._pending) >= self._batch_size
        if batch_full:
            self._safe_flush()

    def flush(self) -> None:
        """Push pending spans to the sink. Raises on sink failure, but the
        batch is re-queued first, so no spans are lost to a transient error."""
        with self._lock:
            if not self._pending:
                return
            batch, self._pending = self._pending, []
            try:
                self._sink.emit(batch)
                self._sink.flush()
            except BaseException:
                self._pending = batch + self._pending
                raise

    def _safe_flush(self) -> None:
        """Flush without raising — used on span-emit and exit paths, where a
        sink error must not replace the user's in-flight exception."""
        try:
            self.flush()
        except Exception:
            logger.warning(
                "stepcost: sink flush failed; %d spans re-queued for retry",
                len(self._pending),
                exc_info=True,
            )

    def close(self) -> None:
        """Flush and release the sink. Raises if the final flush fails."""
        try:
            self.flush()
        finally:
            close = getattr(self._sink, "close", None)
            if callable(close):
                close()
            _LIVE_CLIENTS.discard(self)

    def _require_rollup(self, trace_id: str) -> TraceRollup:
        self.flush()
        rollup = self._trace_rollups.get(trace_id)
        if rollup is None:
            raise TraceCostUnavailableError(
                f"No rollup for trace {trace_id!r}. Start instrumentation with "
                f"`with cc.trace(...) as trace:` on this StepCost instance."
            )
        return rollup

    def trace_total_usd(self, trace_id: str) -> float:
        return float(self._require_rollup(trace_id).total_usd)

    def trace_by_kind(self, trace_id: str) -> dict[str, float]:
        rollup = self._require_rollup(trace_id)
        return {k: float(v) for k, v in rollup.by_kind.items()}

    def trace_by_step(self, trace_id: str) -> dict[str, float]:
        rollup = self._require_rollup(trace_id)
        return {k: float(v) for k, v in rollup.by_step.items()}

    def _trace_total_usd(self, trace_id: str) -> float:
        """Backward-compatible alias."""
        return self.trace_total_usd(trace_id)
