"""Sink protocol for span emission."""

from __future__ import annotations

from typing import Protocol

from stepcost.models import Span


class SpanSink(Protocol):
    def emit(self, spans: list[Span]) -> None: ...

    def flush(self) -> None: ...
