"""Cost-as-Code (beta) — diff two run databases and gate on a budget.

The Infracost pattern for LLM agents: run your instrumented agent suite on the
base branch and on the PR branch, then

    stepcost diff base.db head.db --budget .stepcost.toml --markdown

prints the cost delta (total, per feature, per step) and exits non-zero if the
budget is exceeded — wire it into CI and post the markdown as a PR comment.

Budgets live in `.stepcost.toml` (stdlib tomllib, no new dependency):

    [budget]
    max_total_usd = 5.00        # absolute ceiling for the head run
    max_trace_usd = 0.25        # ceiling for any single trace
    max_increase_pct = 20.0     # head may cost at most base * (1 + pct/100)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from stepcost.report import group_by_trace, load_spans, rollup_by_attr, rollup_by_step, trace_total

_ZERO = Decimal("0")


@dataclass
class Budget:
    max_total_usd: Decimal | None = None
    max_trace_usd: Decimal | None = None
    max_increase_pct: float | None = None

    @classmethod
    def load(cls, path: Path) -> Budget:
        data = tomllib.loads(path.read_text()).get("budget", {})
        dec = lambda k: Decimal(str(data[k])) if k in data else None  # noqa: E731
        return cls(
            max_total_usd=dec("max_total_usd"),
            max_trace_usd=dec("max_trace_usd"),
            max_increase_pct=float(data["max_increase_pct"]) if "max_increase_pct" in data else None,
        )


@dataclass
class CostDiff:
    base_total: Decimal
    head_total: Decimal
    by_feature: dict[str, tuple[Decimal, Decimal]]  # name -> (base, head)
    by_step: dict[str, tuple[Decimal, Decimal]]
    max_trace_usd: Decimal
    violations: list[str] = field(default_factory=list)

    @property
    def delta(self) -> Decimal:
        return self.head_total - self.base_total

    @property
    def pct(self) -> float:
        if self.base_total == 0:
            return 0.0 if self.head_total == 0 else float("inf")
        return float(self.delta / self.base_total * 100)

    @property
    def passed(self) -> bool:
        return not self.violations


def _pair(base: dict[str, Decimal], head: dict[str, Decimal]) -> dict[str, tuple[Decimal, Decimal]]:
    return {
        k: (base.get(k, _ZERO), head.get(k, _ZERO))
        for k in sorted(set(base) | set(head))
    }


def compute_diff(base_db: str, head_db: str, budget: Budget | None = None) -> CostDiff:
    base_spans = load_spans(base_db)
    head_spans = load_spans(head_db)

    head_traces = group_by_trace(head_spans)
    max_trace = max((trace_total(ss) for ss in head_traces.values()), default=_ZERO)

    diff = CostDiff(
        base_total=trace_total(base_spans),
        head_total=trace_total(head_spans),
        by_feature=_pair(
            rollup_by_attr(base_spans, "feature_id"), rollup_by_attr(head_spans, "feature_id")
        ),
        by_step=_pair(rollup_by_step(base_spans), rollup_by_step(head_spans)),
        max_trace_usd=max_trace,
    )

    if budget:
        if budget.max_total_usd is not None and diff.head_total > budget.max_total_usd:
            diff.violations.append(
                f"total ${diff.head_total:.4f} exceeds max_total_usd ${budget.max_total_usd}"
            )
        if budget.max_trace_usd is not None and max_trace > budget.max_trace_usd:
            diff.violations.append(
                f"most expensive trace ${max_trace:.4f} exceeds max_trace_usd ${budget.max_trace_usd}"
            )
        if (
            budget.max_increase_pct is not None
            and diff.base_total > 0
            and diff.pct > budget.max_increase_pct
        ):
            diff.violations.append(
                f"cost increased {diff.pct:.1f}% vs base (max_increase_pct {budget.max_increase_pct}%)"
            )
    return diff


def _arrow(base: Decimal, head: Decimal) -> str:
    if head > base:
        return "🔺"
    if head < base:
        return "🟢"
    return "•"


def render_markdown(diff: CostDiff) -> str:
    sign = "+" if diff.delta >= 0 else ""
    pct = "n/a" if diff.pct == float("inf") else f"{sign}{diff.pct:.1f}%"
    verdict = (
        "✅ **within budget**"
        if diff.passed
        else "❌ **budget exceeded**\n" + "".join(f"\n- {v}" for v in diff.violations)
    )
    lines = [
        "## 💸 StepCost report",
        "",
        f"**Total: ${diff.head_total:.4f}** ({sign}${diff.delta:.4f} vs base, {pct}) — {verdict}",
        "",
        "| Feature | Base | PR | Δ |",
        "|---|---:|---:|:--|",
    ]
    for name, (b, h) in sorted(diff.by_feature.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True):
        lines.append(f"| {name} | ${b:.4f} | ${h:.4f} | {_arrow(b, h)} ${h - b:+.4f} |")
    lines += ["", "| Step | Base | PR | Δ |", "|---|---:|---:|:--|"]
    for name, (b, h) in sorted(diff.by_step.items(), key=lambda kv: kv[1][1] - kv[1][0], reverse=True)[:10]:
        lines.append(f"| {name} | ${b:.4f} | ${h:.4f} | {_arrow(b, h)} ${h - b:+.4f} |")
    lines += [
        "",
        f"<sub>Most expensive single trace: ${diff.max_trace_usd:.4f} · "
        "invoice-grade pricing (cache TTLs, reasoning tokens) · "
        "[stepcost](https://stepcost.com) beta</sub>",
    ]
    return "\n".join(lines)
