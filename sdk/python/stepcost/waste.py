"""Waste-signal heuristics — Rung 2 ("diagnose").

These are FLAGS, not prescriptions. Each signal is a low-effort heuristic over the
span graph, clearly labeled as an estimate. See ROADMAP.md (capability ladder):
this deliberately points at a *likely* leak; it does not yet prescribe the fix
(that is Rung 3 — Recommend).

Estimated dollars are "over the observed traffic in this DB", NOT a monthly
projection — we cannot know production volume from spans alone. A caller may pass
``monthly_multiplier`` to project, but the default (1.0) reports observed waste.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from stepcost.models import PriceTable, Span, SpanKind
from stepcost.pricing import default_price_table, lookup_model

_ZERO = Decimal("0")
_MILLION = Decimal("1000000")

# Heuristic thresholds (intentionally conservative; tune with real dogfood data).
_CACHE_MIN_CALLS = 3
_CACHE_MIN_PREFIX_TOKENS = 1000
_LOOP_MIN_REPEATS = 4  # 4+ repeats of the SAME step ≈ a loop, not normal multi-step
_CONTEXT_MIN_CALLS = 3
_CONTEXT_MIN_LAST_TOKENS = 4000
_FRONTIER_INPUT_RATE = Decimal("2.5")  # >= this $/1M input ≈ frontier-tier model
_TINY_OUTPUT_TOKENS = 200
_CHEAP_INPUT_RATE = Decimal("0.15")  # ~gpt-4o-mini / haiku tier
_CHEAP_OUTPUT_RATE = Decimal("0.60")


@dataclass
class WasteSignal:
    code: str  # missing_cache | retry_loop | oversized_context | model_oversized
    severity: str  # low | medium | high
    message: str
    est_usd: float | None = None  # estimated wasted spend over observed traffic
    span_ids: list[str] = field(default_factory=list)


def _by_trace(spans: list[Span]) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in spans:
        out.setdefault(s.trace_id, []).append(s)
    return out


def detect_waste(
    spans: list[Span],
    *,
    monthly_multiplier: float = 1.0,
    table: PriceTable | None = None,
) -> list[WasteSignal]:
    """Run all heuristics per-trace and return signals ranked by estimated leak."""
    table = table or default_price_table()
    signals: list[WasteSignal] = []
    for group in _by_trace(spans).values():
        signals += _missing_cache(group, monthly_multiplier, table)
        signals += _retry_loop(group, monthly_multiplier)
        signals += _oversized_context(group)
        signals += _model_oversized(group, monthly_multiplier, table)

    sev_rank = {"high": 0, "medium": 1, "low": 2}
    signals.sort(key=lambda w: (-(w.est_usd or 0.0), sev_rank.get(w.severity, 3)))
    return signals


def _missing_cache(group, mult, table) -> list[WasteSignal]:
    by_model: dict[str, list[Span]] = defaultdict(list)
    for s in group:
        if s.kind == SpanKind.LLM_GENERATION and s.model:
            by_model[s.model].append(s)

    out: list[WasteSignal] = []
    for model, calls in by_model.items():
        big = [
            c
            for c in calls
            if c.usage.input_tokens >= _CACHE_MIN_PREFIX_TOKENS
            # Truly uncached only: a call that reads OR writes cache is already
            # using prompt caching — flagging it says "add caching" to someone
            # who has it, the fastest way to lose credibility.
            and c.usage.cached_input_tokens == 0
            and c.usage.cache_creation_tokens == 0
            and c.usage.cache_creation_1h_tokens == 0
        ]
        if len(big) < _CACHE_MIN_CALLS:
            continue
        pricing = lookup_model(model, table)
        if pricing is None or pricing.cached_input_per_1m is None:
            continue  # no cache pricing → can't estimate
        # Assume the shared cacheable prefix ≈ the smallest input seen.
        prefix = min(c.usage.input_tokens for c in big)
        reusable = len(big) - 1  # first call warms the cache
        saving_rate = pricing.input_per_1m - pricing.cached_input_per_1m
        # The warming call pays the write premium (e.g. 1.25x input on
        # Anthropic; 0 on providers without a write charge).
        write_premium = (
            pricing.cache_write_per_1m - pricing.input_per_1m
            if pricing.cache_write_per_1m is not None
            else Decimal("0")
        )
        est = (
            float(
                (Decimal(prefix * reusable) * saving_rate - Decimal(prefix) * write_premium)
                / _MILLION
            )
            * mult
        )
        if est <= 0:
            continue
        out.append(
            WasteSignal(
                code="missing_cache",
                severity="high" if est >= 50 else "medium",
                message=(
                    f"{len(big)} uncached {model} calls share a ~{prefix:,}-token prefix — "
                    f"prompt caching likely cuts the repeats."
                ),
                est_usd=round(est, 4),
                span_ids=[c.span_id for c in big],
            )
        )
    return out


def _nearest_step_name(span: Span, by_id: dict[str, Span]) -> str | None:
    """Name of the closest agent_step ancestor, for cost attribution."""
    seen: set[str] = set()
    parent_id = span.parent_span_id
    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = by_id.get(parent_id)
        if parent is None:
            return None
        if parent.kind == SpanKind.AGENT_STEP:
            return parent.name
        parent_id = parent.parent_span_id
    return None


def _retry_loop(group, mult) -> list[WasteSignal]:
    """One signal per agent_step name that repeats like a loop.

    We key on the *agent_step* (the loop structure), not on model/tool — otherwise a
    normal 4-iteration loop emits a flag for every span kind inside it. Cost is
    attributed to the step's descendants so the estimate reflects the repeated work.
    """
    by_id = {s.span_id: s for s in group}
    step_spans = [s for s in group if s.kind == SpanKind.AGENT_STEP and s.name]
    counts = Counter(s.name for s in step_spans)

    cost_by_step: dict[str, Decimal] = defaultdict(lambda: _ZERO)
    for s in group:
        if s.kind in (SpanKind.TRACE, SpanKind.AGENT_STEP):
            continue
        if s.cost.total_usd == _ZERO:
            continue
        name = _nearest_step_name(s, by_id)
        if name:
            cost_by_step[name] += s.cost.total_usd

    out: list[WasteSignal] = []
    for name, count in counts.items():
        if count < _LOOP_MIN_REPEATS:
            continue
        total = cost_by_step.get(name, _ZERO)
        if total == _ZERO:
            continue  # a free step looping isn't a *cost* leak — don't flag noise
        est = float(total * (count - 1) / count) * mult if count else 0.0
        out.append(
            WasteSignal(
                code="retry_loop",
                severity="high" if count >= 8 else "medium",
                message=f"'{name}' step ran {count}× in one trace — possible runaway loop.",
                est_usd=round(est, 4) if est > 0 else None,
                span_ids=[s.span_id for s in step_spans if s.name == name],
            )
        )
    return out


def _oversized_context(group) -> list[WasteSignal]:
    llm = sorted(
        (s for s in group if s.kind == SpanKind.LLM_GENERATION),
        key=lambda s: s.started_at,
    )
    if len(llm) < _CONTEXT_MIN_CALLS:
        return []
    inputs = [s.usage.input_tokens for s in llm]
    first, last = inputs[0], inputs[-1]
    monotonic = all(inputs[i] <= inputs[i + 1] for i in range(len(inputs) - 1))
    if monotonic and first > 0 and last >= 2 * first and last >= _CONTEXT_MIN_LAST_TOKENS:
        return [
            WasteSignal(
                code="oversized_context",
                severity="medium",
                message=(
                    f"Input grew {first:,}→{last:,} tokens across {len(llm)} calls — "
                    f"full history likely re-sent; consider trimming/summarizing."
                ),
                est_usd=None,  # savings depend on how much history is trimmable
                span_ids=[s.span_id for s in llm],
            )
        ]
    return []


def _model_oversized(group, mult, table) -> list[WasteSignal]:
    candidates: dict[str, list[tuple[Span, float]]] = defaultdict(list)
    for s in group:
        if s.kind != SpanKind.LLM_GENERATION or not s.model:
            continue
        pricing = lookup_model(s.model, table)
        if pricing is None:
            continue
        if pricing.input_per_1m < _FRONTIER_INPUT_RATE:
            continue
        if not (0 < s.usage.output_tokens < _TINY_OUTPUT_TOKENS):
            continue
        cheap = (Decimal(s.usage.input_tokens) / _MILLION) * _CHEAP_INPUT_RATE + (
            Decimal(s.usage.output_tokens) / _MILLION
        ) * _CHEAP_OUTPUT_RATE
        est = float(s.cost.total_usd - cheap) * mult
        if est <= 0:
            continue
        candidates[s.model].append((s, est))

    out: list[WasteSignal] = []
    for model, items in candidates.items():
        total_est = round(sum(e for _s, e in items), 4)
        out.append(
            WasteSignal(
                code="model_oversized",
                severity="low",
                message=(
                    f"{len(items)} {model} call(s) produced tiny outputs "
                    f"(<{_TINY_OUTPUT_TOKENS} tok) — a cheaper model may suffice (low confidence)."
                ),
                est_usd=total_est,
                span_ids=[s.span_id for s, _e in items],
            )
        )
    return out
