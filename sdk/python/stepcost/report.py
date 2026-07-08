"""Read spans from a StepCost sink DB and build the agent-economics report.

Pure data layer — no rendering (``cli.py`` renders these structures). Reconstructs
full ``Span`` objects from the persisted payload JSON, rebuilds the trace tree via
``parent_span_id``, and computes the rollups that make up Pillar A (the typed
agent-economics graph).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from stepcost.models import Span, SpanKind
from stepcost.sinks.sqlite import sqlite_url_to_path
from stepcost.sync import ProviderCost
from stepcost.waste import WasteSignal, detect_waste

_ZERO = Decimal("0")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def resolve_db_path(url_or_path: str) -> Path:
    """Accept a plain path or a ``sqlite:///path`` URL (mirrors SQLiteSink.from_url)."""
    if url_or_path.startswith("sqlite:"):
        return sqlite_url_to_path(url_or_path)
    return Path(url_or_path).expanduser()


def load_spans(url_or_path: str, *, trace_id: str | None = None) -> list[Span]:
    db_path = resolve_db_path(url_or_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No StepCost DB at {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        if trace_id:
            rows = conn.execute(
                "SELECT payload_json FROM spans WHERE trace_id = ?", (trace_id,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT payload_json FROM spans").fetchall()
    finally:
        conn.close()
    return [Span.model_validate(json.loads(row[0])) for row in rows]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def group_by_trace(spans: list[Span]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in spans:
        out.setdefault(s.trace_id, []).append(s)
    return out


def _index(spans: list[Span]) -> dict[str, Span]:
    return {s.span_id: s for s in spans}


def _nearest_agent_step(span: Span, by_id: dict[str, Span]) -> Span | None:
    """Walk up parents to the closest agent_step ancestor (for by_step attribution)."""
    seen: set[str] = set()
    parent_id = span.parent_span_id
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return None
        if parent.kind == SpanKind.AGENT_STEP:
            return parent
        parent_id = parent.parent_span_id
    return None


def trace_total(spans: list[Span]) -> Decimal:
    # agent_step / trace spans carry no cost, so summing non-trace spans is safe.
    return sum((s.cost.total_usd for s in spans if s.kind != SpanKind.TRACE), _ZERO)


def rollup_by_kind(spans: list[Span]) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for s in spans:
        if s.kind == SpanKind.TRACE:
            continue
        out[s.kind.value] = out.get(s.kind.value, _ZERO) + s.cost.total_usd
    return out


def rollup_by_step(spans: list[Span]) -> dict[str, Decimal]:
    by_id = _index(spans)
    out: dict[str, Decimal] = {}
    for s in spans:
        if s.kind in (SpanKind.TRACE, SpanKind.AGENT_STEP):
            continue
        if s.cost.total_usd == _ZERO:
            continue
        step = _nearest_agent_step(s, by_id)
        name = step.name if step and step.name else "(no step)"
        out[name] = out.get(name, _ZERO) + s.cost.total_usd
    return out


def unpriced_models(spans: list[Span]) -> dict[str, int]:
    """Models whose spans carry real tokens but no price-table match ($0 cost).

    Surfaced loudly in the report: silently missing a model from the price
    table would otherwise undercount the total with no visible signal.
    """
    out: dict[str, int] = {}
    for s in spans:
        if s.model and s.usage.total_tokens > 0 and s.cost.price_table_version is None:
            out[s.model] = out.get(s.model, 0) + 1
    return out


def rollup_by_attr(spans: list[Span], attr: str) -> dict[str, Decimal]:
    out: dict[str, Decimal] = {}
    for s in spans:
        if s.kind == SpanKind.TRACE:
            continue
        key = getattr(s, attr) or "(unset)"
        out[key] = out.get(key, _ZERO) + s.cost.total_usd
    return out


# --------------------------------------------------------------------------- #
# Tree
# --------------------------------------------------------------------------- #
@dataclass
class TreeNode:
    span: Span
    children: list[TreeNode] = field(default_factory=list)
    subtree_usd: Decimal = _ZERO


def build_tree(spans: list[Span]) -> list[TreeNode]:
    by_id = _index(spans)
    children_map: dict[str | None, list[Span]] = {}
    for s in spans:
        children_map.setdefault(s.parent_span_id, []).append(s)

    def make(span: Span) -> TreeNode:
        node = TreeNode(span=span)
        kids = sorted(children_map.get(span.span_id, []), key=lambda x: x.started_at)
        node.children = [make(k) for k in kids]
        node.subtree_usd = span.cost.total_usd + sum(
            (c.subtree_usd for c in node.children), _ZERO
        )
        return node

    # A span is a root if it has no parent, or its parent isn't in this span set.
    roots = [s for s in spans if s.parent_span_id is None or s.parent_span_id not in by_id]
    return [make(r) for r in sorted(roots, key=lambda x: x.started_at)]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class TraceReport:
    trace_id: str
    total_usd: Decimal
    n_spans: int
    feature_id: str | None
    customer_id: str | None
    roots: list[TreeNode]
    by_kind: dict[str, Decimal]
    by_step: dict[str, Decimal]
    waste: list[WasteSignal]
    unpriced: dict[str, int] = field(default_factory=dict)


def build_trace_report(
    trace_id: str, spans: list[Span], *, monthly_multiplier: float = 1.0
) -> TraceReport:
    return TraceReport(
        trace_id=trace_id,
        total_usd=trace_total(spans),
        n_spans=len(spans),
        feature_id=next((s.feature_id for s in spans if s.feature_id), None),
        customer_id=next((s.customer_id for s in spans if s.customer_id), None),
        roots=build_tree(spans),
        by_kind=rollup_by_kind(spans),
        by_step=rollup_by_step(spans),
        waste=detect_waste(spans, monthly_multiplier=monthly_multiplier),
        unpriced=unpriced_models(spans),
    )


@dataclass
class ReconciliationRow:
    """SDK-observed vs provider-billed dollars for one provider-day."""

    provider: str
    day: str  # YYYY-MM-DD UTC
    sdk_usd: Decimal
    provider_usd: Decimal

    @property
    def drift_pct(self) -> float:
        if self.provider_usd == 0:
            return 0.0
        return float(abs(self.sdk_usd - self.provider_usd) / self.provider_usd * 100)

    @property
    def coverage_pct(self) -> float:
        """How much of the provider's bill the SDK saw (uninstrumented spend gap)."""
        if self.provider_usd == 0:
            return 100.0
        return float(self.sdk_usd / self.provider_usd * 100)


def build_reconciliation(
    spans: list[Span], provider_costs: list[ProviderCost]
) -> list[ReconciliationRow]:
    """One row per provider-day the provider reported costs for.

    The provider ledger is treated as ground truth; the SDK side is what got
    instrumented. drift ≈ pricing accuracy; coverage < 100% ≈ spend the SDK
    never saw (uninstrumented call sites).
    """
    billed: dict[tuple[str, str], Decimal] = {}
    for pc in provider_costs:
        key = (pc.provider, pc.day)
        billed[key] = billed.get(key, _ZERO) + pc.amount_usd

    observed: dict[tuple[str, str], Decimal] = {}
    for s in spans:
        if s.kind == SpanKind.TRACE or s.provider is None or s.cost.total_usd == _ZERO:
            continue
        key = (s.provider.value, s.started_at.strftime("%Y-%m-%d"))
        observed[key] = observed.get(key, _ZERO) + s.cost.total_usd

    return [
        ReconciliationRow(provider=p, day=d, sdk_usd=observed.get((p, d), _ZERO), provider_usd=usd)
        for (p, d), usd in sorted(billed.items())
        if usd > 0
    ]


@dataclass
class Summary:
    total_usd: Decimal
    n_traces: int
    n_spans: int
    by_feature: dict[str, Decimal]
    by_customer: dict[str, Decimal]
    by_kind: dict[str, Decimal]
    top_traces: list[tuple[str, Decimal]]
    waste: list[WasteSignal]
    unpriced: dict[str, int] = field(default_factory=dict)
    reconciliation: list[ReconciliationRow] = field(default_factory=list)


def build_summary(
    spans: list[Span],
    *,
    top: int = 5,
    monthly_multiplier: float = 1.0,
    provider_costs: list[ProviderCost] | None = None,
) -> Summary:
    grouped = group_by_trace(spans)
    top_traces = sorted(
        ((tid, trace_total(ss)) for tid, ss in grouped.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:top]
    return Summary(
        total_usd=trace_total(spans),
        n_traces=len(grouped),
        n_spans=len(spans),
        by_feature=rollup_by_attr(spans, "feature_id"),
        by_customer=rollup_by_attr(spans, "customer_id"),
        by_kind=rollup_by_kind(spans),
        top_traces=top_traces,
        waste=detect_waste(spans, monthly_multiplier=monthly_multiplier),
        unpriced=unpriced_models(spans),
        reconciliation=build_reconciliation(spans, provider_costs or []),
    )
