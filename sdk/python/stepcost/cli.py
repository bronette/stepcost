"""`stepcost` CLI — render the agent-economics graph from a sink DB.

Usage:
    stepcost report <db>                 # cost summary + waste flags across all traces
    stepcost report <db> --trace <id>    # the typed cost tree for one trace
    stepcost report <db> --top 10        # top-N most expensive traces
    stepcost report <db> --json          # machine-readable output
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

from stepcost import report as R
from stepcost.waste import WasteSignal


def _usd(value: Decimal | float) -> str:
    return f"${float(value):,.4f}"


def _fmt_rows(mapping: dict[str, Decimal], indent: str = "  ") -> list[str]:
    if not mapping:
        return [f"{indent}(none)"]
    rows = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)
    return [f"{indent}{k:<28} {_usd(v)}" for k, v in rows]


def _node_label(span) -> str:
    kind = span.kind.value
    if kind == "trace":
        return f"trace {span.name or ''}".rstrip()
    if kind == "agent_step":
        return f"agent_step: {span.name or '?'}"
    if kind == "llm_generation":
        return f"llm {span.model or span.name or '?'}"
    return f"{kind} {span.name or ''}".rstrip()


def _fmt_tokens(n: int) -> str:
    return f"{n:,} tok" if n else "—"


def _subtree_tokens(node: R.TreeNode) -> int:
    own = node.span.usage.total_tokens if node.span.usage else 0
    return own + sum(_subtree_tokens(c) for c in node.children)


def _render_tree(node: R.TreeNode, lines: list[str], prefix: str = "", is_last: bool = True, root: bool = True) -> None:
    toks = _fmt_tokens(_subtree_tokens(node))
    if root:
        lines.append(f"{_node_label(node.span):<44} {toks:>12}  {_usd(node.subtree_usd)}")
        child_prefix = ""
    else:
        connector = "└─ " if is_last else "├─ "
        label = f"{prefix}{connector}{_node_label(node.span)}"
        lines.append(f"{label:<44} {toks:>12}  {_usd(node.subtree_usd)}")
        child_prefix = prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(node.children):
        _render_tree(child, lines, child_prefix, i == len(node.children) - 1, root=False)


def _fmt_rows_or_tokens(usd: dict[str, Decimal], tokens: dict[str, int], indent: str = "  ") -> list[str]:
    """Dollar rows normally; token rows for $0 workloads (local Ollama etc.),
    so a free run still shows its real step/kind breakdown."""
    if any(v > 0 for v in usd.values()) or not tokens:
        return _fmt_rows(usd)
    rows = sorted(tokens.items(), key=lambda kv: kv[1], reverse=True)
    return [f"{indent}{k:<28} {v:,} tok" for k, v in rows]


def _render_unpriced(unpriced: dict[str, int]) -> list[str]:
    if not unpriced:
        return []
    lines = ["", "⚠ Unpriced spans (recorded $0 — total is an undercount):"]
    for model, count in sorted(unpriced.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {model}: {count} span(s) with no price-table entry")
    lines.append("  Fix: update price_table.json or register_custom_model().")
    return lines


def _render_reconciliation(rows: list) -> list[str]:
    if not rows:
        return []
    lines = ["", "Provider reconciliation (SDK-observed vs provider-billed):"]
    for r in rows:
        lines.append(
            f"  {r.provider:<10} {r.day}   SDK {_usd(r.sdk_usd)}   "
            f"billed {_usd(r.provider_usd)}   drift {r.drift_pct:.1f}%   "
            f"coverage {r.coverage_pct:.0f}%"
        )
    worst = max(rows, key=lambda r: r.drift_pct)
    gate = "PASS ✅" if worst.drift_pct <= 2.0 else "CHECK ⚠"
    lines.append(f"  2% gate (worst day): {worst.drift_pct:.2f}% — {gate}")
    lines.append("  coverage < 100% = spend the SDK never saw (uninstrumented call sites)")
    return lines


def _render_waste(waste: list[WasteSignal]) -> list[str]:
    if not waste:
        return ["  (no waste flags)"]
    out = []
    for w in waste:
        est = f" ~{_usd(w.est_usd)}" if w.est_usd else ""
        out.append(f"  [{w.severity}] {w.code}{est} — {w.message}")
    out.append("  (heuristic flags over observed traffic — verify before acting)")
    return out


def _cmd_report(args: argparse.Namespace) -> int:
    spans = R.load_spans(args.db, trace_id=args.trace)
    if not spans:
        print("No spans found." + (f" (trace {args.trace})" if args.trace else ""))
        return 1

    if args.trace:
        rep = R.build_trace_report(args.trace, spans, monthly_multiplier=args.multiplier)
        if args.json:
            print(json.dumps(_trace_json(rep), default=str, indent=2))
            return 0
        print(f"Trace {rep.trace_id}")
        meta = " / ".join(
            p for p in [
                f"feature={rep.feature_id}" if rep.feature_id else "",
                f"customer={rep.customer_id}" if rep.customer_id else "",
            ] if p
        )
        print(f"{meta}   {_usd(rep.total_usd)}   ({rep.n_spans} spans)\n")
        tree_lines: list[str] = []
        for node in rep.roots:
            _render_tree(node, tree_lines)
        print("\n".join(tree_lines))
        print("\nBy step:")
        print("\n".join(_fmt_rows_or_tokens(rep.by_step, rep.by_step_tokens)))
        print("\nBy kind:")
        print("\n".join(_fmt_rows_or_tokens(rep.by_kind, rep.by_kind_tokens)))
        print("\nWaste signals:")
        print("\n".join(_render_waste(rep.waste)))
        for line in _render_unpriced(rep.unpriced):
            print(line)
        return 0

    from stepcost.sync import load_provider_costs

    provider_costs = load_provider_costs(R.resolve_db_path(args.db))
    summary = R.build_summary(
        spans, top=args.top, monthly_multiplier=args.multiplier, provider_costs=provider_costs
    )
    if args.html:
        from stepcost.report_html import render_html

        grouped = R.group_by_trace(spans)
        traces = [
            R.build_trace_report(tid, grouped[tid], monthly_multiplier=args.multiplier)
            for tid, _usd_ in summary.top_traces
            if tid in grouped
        ]
        out = Path(args.html).expanduser()
        out.write_text(render_html(summary, traces, source=args.db))
        print(f"wrote {out}  (open it in a browser)")
        return 0
    if args.json:
        print(json.dumps(_summary_json(summary), default=str, indent=2))
        return 0
    print(f"StepCost report — {args.db}")
    print(f"Total: {_usd(summary.total_usd)}  across {summary.n_traces} traces / {summary.n_spans} spans\n")
    print("By feature:")
    print("\n".join(_fmt_rows(summary.by_feature)))
    print("\nBy customer:")
    print("\n".join(_fmt_rows(summary.by_customer)))
    print("\nBy kind:")
    print("\n".join(_fmt_rows(summary.by_kind)))
    print(f"\nTop {args.top} traces:")
    if summary.top_traces:
        for tid, usd in summary.top_traces:
            print(f"  {tid[:12]:<14} {_usd(usd)}")
    else:
        print("  (none)")
    print("\nWaste signals:")
    print("\n".join(_render_waste(summary.waste)))
    for line in _render_reconciliation(summary.reconciliation):
        print(line)
    for line in _render_unpriced(summary.unpriced):
        print(line)
    return 0


def _waste_json(waste: list[WasteSignal]) -> list[dict]:
    return [
        {"code": w.code, "severity": w.severity, "message": w.message,
         "est_usd": w.est_usd, "span_ids": w.span_ids}
        for w in waste
    ]


def _tree_json(node: R.TreeNode) -> dict:
    span = node.span
    return {
        "span_id": span.span_id,
        "kind": span.kind.value,
        "name": span.name,
        "model": span.model,
        "total_tokens": span.usage.total_tokens if span.usage else 0,
        "cost_usd": str(span.cost.total_usd),
        "subtree_usd": str(node.subtree_usd),
        "children": [_tree_json(c) for c in node.children],
    }


def _trace_json(rep: R.TraceReport) -> dict:
    return {
        "trace_id": rep.trace_id,
        "total_usd": str(rep.total_usd),
        "n_spans": rep.n_spans,
        "feature_id": rep.feature_id,
        "customer_id": rep.customer_id,
        "tree": [_tree_json(r) for r in rep.roots],
        "by_step": {k: str(v) for k, v in rep.by_step.items()},
        "by_kind": {k: str(v) for k, v in rep.by_kind.items()},
        "waste": _waste_json(rep.waste),
        "unpriced": rep.unpriced,
    }


def _summary_json(s: R.Summary) -> dict:
    return {
        "total_usd": str(s.total_usd),
        "n_traces": s.n_traces,
        "n_spans": s.n_spans,
        "by_feature": {k: str(v) for k, v in s.by_feature.items()},
        "by_customer": {k: str(v) for k, v in s.by_customer.items()},
        "by_kind": {k: str(v) for k, v in s.by_kind.items()},
        "top_traces": [[tid, str(usd)] for tid, usd in s.top_traces],
        "waste": _waste_json(s.waste),
        "unpriced": s.unpriced,
        "reconciliation": [
            {
                "provider": r.provider,
                "day": r.day,
                "sdk_usd": str(r.sdk_usd),
                "provider_usd": str(r.provider_usd),
                "drift_pct": round(r.drift_pct, 3),
                "coverage_pct": round(r.coverage_pct, 1),
            }
            for r in s.reconciliation
        ],
    }


def _cmd_sync(args: argparse.Namespace) -> int:
    import os

    from stepcost import sync as S

    db_path = R.resolve_db_path(args.db)
    a_start, a_end, o_start, o_end = S.default_window(args.days)

    if args.provider == "anthropic":
        key = os.environ.get("ANTHROPIC_ADMIN_KEY", "")
        if not key.startswith("sk-ant-admin"):
            print(
                "error: set ANTHROPIC_ADMIN_KEY to an Admin API key (sk-ant-admin01-...).\n"
                "Create one: Console → Settings → Organization → Admin keys. "
                "A regular sk-ant-api key cannot read org cost reports.",
                file=sys.stderr,
            )
            return 2
        records = S.fetch_anthropic_costs(key, starting_at=a_start, ending_at=a_end)
    else:
        key = os.environ.get("OPENAI_ADMIN_KEY", "")
        if not key:
            print(
                "error: set OPENAI_ADMIN_KEY to an org admin key "
                "(platform.openai.com → Settings → Organization → Admin keys).",
                file=sys.stderr,
            )
            return 2
        records = S.fetch_openai_costs(key, start_time=o_start, end_time=o_end)

    n = S.store_provider_costs(db_path, records)
    total = sum((r.amount_usd for r in records), start=Decimal("0"))
    print(f"synced {n} {args.provider} cost rows (last {args.days}d, ${total:.4f}) → {db_path}")
    print("Run `stepcost report` to see the reconciliation section.")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    from stepcost.diff import Budget, compute_diff, render_markdown

    budget = None
    if args.budget:
        budget_path = Path(args.budget).expanduser()
        if not budget_path.exists():
            print(f"error: budget file not found: {budget_path}", file=sys.stderr)
            return 2
        budget = Budget.load(budget_path)
    diff = compute_diff(args.base_db, args.head_db, budget)
    if args.markdown:
        print(render_markdown(diff))
    else:
        sign = "+" if diff.delta >= 0 else ""
        print(f"base:  {_usd(diff.base_total)}")
        print(f"head:  {_usd(diff.head_total)}  ({sign}{_usd(diff.delta)}, "
              f"{'n/a' if diff.pct == float('inf') else f'{sign}{diff.pct:.1f}%'})")
        for v in diff.violations:
            print(f"BUDGET: {v}", file=sys.stderr)
        print("budget:", "PASS" if diff.passed else "FAIL")
    return 0 if diff.passed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stepcost", description="FinOps for LLM agents")
    sub = parser.add_subparsers(dest="command", required=True)

    sync_p = sub.add_parser(
        "sync", help="pull provider-billed costs (org admin APIs) into the sink DB"
    )
    sync_p.add_argument("provider", choices=["anthropic", "openai"])
    sync_p.add_argument("db", help="path or sqlite:/// URL to a StepCost SQLite sink")
    sync_p.add_argument("--days", type=int, default=7, help="how many days back to pull")
    sync_p.set_defaults(func=_cmd_sync)

    dif = sub.add_parser("diff", help="(beta) cost delta between two sink DBs; gate on a budget")
    dif.add_argument("base_db", help="baseline run DB (e.g. main branch)")
    dif.add_argument("head_db", help="head run DB (e.g. this PR)")
    dif.add_argument("--budget", default=None, help="path to .stepcost.toml with a [budget] table")
    dif.add_argument("--markdown", action="store_true", help="emit a PR-comment-ready markdown body")
    dif.set_defaults(func=_cmd_diff)

    rep = sub.add_parser("report", help="render cost report from a sink DB")
    rep.add_argument("db", help="path or sqlite:/// URL to a StepCost SQLite sink")
    rep.add_argument("--trace", help="render one trace's cost tree", default=None)
    rep.add_argument("--top", type=int, default=5, help="top-N traces in the summary")
    rep.add_argument(
        "--multiplier", type=float, default=1.0,
        help="scale waste $ estimates to project beyond observed traffic",
    )
    rep.add_argument("--json", action="store_true", help="machine-readable output")
    rep.add_argument("--html", metavar="OUT", default=None,
                     help="write a self-contained HTML report (summary view only)")
    rep.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except sqlite3.DatabaseError as exc:
        print(
            f"error: not a readable StepCost database ({exc}). "
            "Point `stepcost report` at a SQLite file written by SQLiteSink.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
