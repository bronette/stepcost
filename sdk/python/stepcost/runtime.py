"""Runtime context — active client and trace for implicit DX."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stepcost.client import StepCost
    from stepcost.context import ActiveSpan, TraceContext

_current_client: contextvars.ContextVar[StepCost | None] = contextvars.ContextVar(
    "stepcost_client", default=None
)
_current_trace: contextvars.ContextVar[TraceContext | None] = contextvars.ContextVar(
    "stepcost_trace", default=None
)
# Immutable tuple, not a shared list: each asyncio task sees its own copy of the
# stack, so concurrent spans under one trace parent correctly (OTel-style).
_span_stack: contextvars.ContextVar[tuple[ActiveSpan, ...]] = contextvars.ContextVar(
    "stepcost_span_stack", default=()
)


def get_client() -> StepCost | None:
    return _current_client.get()


def get_trace() -> TraceContext | None:
    return _current_trace.get()


def set_client(client: StepCost | None) -> contextvars.Token:
    return _current_client.set(client)


def set_trace(trace: TraceContext | None) -> contextvars.Token:
    return _current_trace.set(trace)


def reset_client(token: contextvars.Token) -> None:
    _current_client.reset(token)


def reset_trace(token: contextvars.Token) -> None:
    _current_trace.reset(token)


def current_span() -> ActiveSpan | None:
    stack = _span_stack.get()
    return stack[-1] if stack else None


def push_span(active: ActiveSpan) -> contextvars.Token:
    return _span_stack.set(_span_stack.get() + (active,))


def pop_span(token: contextvars.Token) -> None:
    try:
        _span_stack.reset(token)
    except ValueError:
        # Token from another context (span finished in a different task/thread
        # than it was opened in) — that context's stack copy dies with it.
        pass
