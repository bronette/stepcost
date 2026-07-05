"""In-memory trace rollups — agent economics graph aggregates."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from stepcost.models import Span, SpanKind


@dataclass
class TraceRollup:
    total_usd: Decimal = field(default_factory=lambda: Decimal("0"))
    by_kind: dict[str, Decimal] = field(default_factory=dict)
    by_step: dict[str, Decimal] = field(default_factory=dict)

    def add(self, span: Span, *, agent_step: str | None) -> None:
        if span.kind == SpanKind.TRACE:
            return
        amount = span.cost.total_usd
        self.total_usd += amount
        kind_key = span.kind.value
        self.by_kind[kind_key] = self.by_kind.get(kind_key, Decimal("0")) + amount
        if agent_step:
            self.by_step[agent_step] = self.by_step.get(agent_step, Decimal("0")) + amount

    def as_floats(self) -> tuple[float, dict[str, float], dict[str, float]]:
        return (
            float(self.total_usd),
            {k: float(v) for k, v in self.by_kind.items()},
            {k: float(v) for k, v in self.by_step.items()},
        )
