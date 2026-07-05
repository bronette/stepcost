"""Trace and span context managers."""

from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator
from uuid import uuid4

from stepcost.errors import ContextRequiredError
from stepcost.extractors import extract_usage
from stepcost.models import Provider, Span, SpanKind, TokenUsage
from stepcost.pricing import cost_for_model
from stepcost.runtime import (
    get_client,
    pop_span,
    push_span,
    reset_client,
    reset_trace,
    set_client,
    set_trace,
)

if TYPE_CHECKING:
    from stepcost.client import StepCost

logger = logging.getLogger("stepcost")
_warned_unpriced: set[str] = set()


def _warn_unpriced(model: str) -> None:
    # Once per model per process — an unknown model silently costing $0 is the
    # worst failure mode for a cost-accuracy SDK, so make it loud.
    if model not in _warned_unpriced:
        _warned_unpriced.add(model)
        logger.warning(
            "stepcost: no pricing for model %r — its spans will record $0. "
            "Add it with register_custom_model() or update price_table.json.",
            model,
        )


def _require_client(explicit: StepCost | None = None) -> StepCost:
    client = explicit or get_client()
    if client is None:
        raise ContextRequiredError(
            "No active StepCost client. Use `with cc.trace(...) as trace:` first, "
            "or pass client= explicitly to agent_step / llm_call."
        )
    return client


def _coerce_provider(provider: Provider | str) -> Provider:
    if isinstance(provider, Provider):
        return provider
    return Provider(provider)


@dataclass
class ActiveSpan:
    client: StepCost
    span: Span
    _start: float = field(default_factory=time.perf_counter)
    _finished: bool = False
    _stack_token: contextvars.Token | None = field(default=None, repr=False)

    def record_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        reasoning_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_creation_1h_tokens: int = 0,
        embedding_tokens: int = 0,
    ) -> None:
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            reasoning_tokens=reasoning_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_creation_1h_tokens=cache_creation_1h_tokens,
            embedding_tokens=embedding_tokens,
        )
        self._apply_usage(usage)

    def record(self, response: Any, *, provider: Provider | str | None = None) -> None:
        """Extract invoice-safe token usage from an OpenAI or Anthropic response."""
        prov = _coerce_provider(provider or self.span.provider or Provider.OTHER)
        usage = extract_usage(response, provider=prov)
        self._apply_usage(usage)

    def _apply_usage(self, usage: TokenUsage) -> None:
        self.span.usage = usage
        if self.span.model:
            cost = cost_for_model(self.span.model, usage, table=self.client.price_table)
            if cost is not None:
                self.span.cost = cost
            elif usage.total_tokens > 0:
                _warn_unpriced(self.span.model)

    def set_metadata(self, **tags: str) -> None:
        self.span.tags.update(tags)

    def finish(self) -> Span:
        if self._finished:
            return self.span
        self._finished = True
        self.span.duration_ms = int((time.perf_counter() - self._start) * 1000)
        if self._stack_token is not None:
            pop_span(self._stack_token)
            self._stack_token = None
        self.client._emit(self.span)
        return self.span

    def __enter__(self) -> ActiveSpan:
        self._stack_token = push_span(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish()


@contextmanager
def agent_step(name: str, *, client: StepCost | None = None) -> Iterator[ActiveSpan]:
    cc = _require_client(client)
    with cc.span(kind=SpanKind.AGENT_STEP, name=name) as step:
        yield step


@contextmanager
def llm_call(
    *,
    model: str,
    provider: Provider | str = Provider.OPENAI,
    prompt_version: str | None = None,
    client: StepCost | None = None,
) -> Iterator[ActiveSpan]:
    cc = _require_client(client)
    with cc.span(
        kind=SpanKind.LLM_GENERATION,
        name=model,
        model=model,
        provider=_coerce_provider(provider),
        prompt_version=prompt_version,
    ) as call:
        yield call


class TraceContext:
    def __init__(
        self,
        client: StepCost,
        *,
        customer_id: str | None = None,
        feature_id: str | None = None,
        team_id: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.client = client
        self.trace_id = str(uuid4())
        self.customer_id = customer_id or client.default_customer_id
        self.feature_id = feature_id or client.default_feature
        self.team_id = team_id or client.default_team_id
        self.session_id = session_id
        self.user_id = user_id
        self.name = name
        self._client_token = None
        self._trace_token = None
        self.root_span: Span | None = None
        self._start = time.perf_counter()

    def __enter__(self) -> TraceContext:
        self._client_token = set_client(self.client)
        self._trace_token = set_trace(self)
        self.client._init_trace_rollup(self.trace_id)
        self.root_span = Span(
            span_id=str(uuid4()),
            trace_id=self.trace_id,
            kind=SpanKind.TRACE,
            name=self.name,
            project_id=self.client.project,
            environment=self.client.environment,
            customer_id=self.customer_id,
            feature_id=self.feature_id,
            team_id=self.team_id,
            session_id=self.session_id,
            user_id=self.user_id,
        )
        self.client._index_span(self.root_span)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.root_span is not None:
            self.root_span.duration_ms = int((time.perf_counter() - self._start) * 1000)
            self.client._emit(self.root_span)
        if self._trace_token is not None:
            reset_trace(self._trace_token)
        if self._client_token is not None:
            reset_client(self._client_token)
        # Persist the trace at its boundary so short scripts that never fill a
        # batch (or never call flush()) still land in the sink. Never raises —
        # a sink outage must not clobber an exception unwinding through here.
        self.client._safe_flush()

    @property
    def total_usd(self) -> float:
        return self.client.trace_total_usd(self.trace_id)

    @property
    def by_kind(self) -> dict[str, float]:
        return self.client.trace_by_kind(self.trace_id)

    @property
    def by_step(self) -> dict[str, float]:
        return self.client.trace_by_step(self.trace_id)
