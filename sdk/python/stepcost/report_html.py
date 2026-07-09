"""Self-contained HTML report — `stepcost report <db> --html out.html`.

One dark-theme file, zero external assets, zero JS dependencies (native
<details> for trace trees). Open locally or attach to a PR/slack thread.
"""

from __future__ import annotations

import html
from decimal import Decimal

from stepcost.report import Summary, TraceReport, TreeNode

_CSS = """
:root{--bg:#0b0f14;--surface:#121820;--border:#243041;--text:#e8eef6;--muted:#8fa3bc;
--accent:#3dd68c;--warn:#f5a623;--code:#070a0f}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);
line-height:1.6;padding:2rem;max-width:960px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:.25rem}h1 span{color:var(--accent)}
p.sub{color:var(--muted);margin-bottom:1.6rem;font-size:.92rem}
h2{font-size:1.05rem;margin:1.8rem 0 .6rem;color:var(--text)}
table{width:100%;border-collapse:collapse;font-size:.9rem;background:var(--surface);
border:1px solid var(--border);border-radius:8px}
th,td{text-align:left;padding:.5rem .7rem;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:500}td.usd{text-align:right;font-variant-numeric:tabular-nums}
.kpis{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:.5rem}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;
padding:.8rem 1.1rem;min-width:140px}
.kpi .v{font-size:1.35rem;font-weight:700}.kpi .l{color:var(--muted);font-size:.8rem}
.flag{border-left:3px solid var(--warn);background:var(--surface);border-radius:6px;
padding:.55rem .8rem;margin:.4rem 0;font-size:.9rem}
.ok{color:var(--accent)}.warn{color:var(--warn)}
details{background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:.6rem .9rem;margin:.4rem 0}
summary{cursor:pointer;font-size:.92rem}
pre{font-family:ui-monospace,monospace;font-size:.78rem;line-height:1.55;color:#c9d6e5;
overflow-x:auto;padding:.6rem 0 0}
footer{margin-top:2.5rem;color:var(--muted);font-size:.82rem}
footer a{color:var(--accent);text-decoration:none}
"""


def _usd(v: Decimal | float) -> str:
    return f"${float(v):,.4f}"


def _rows(mapping: dict[str, Decimal]) -> str:
    if not mapping:
        return "<tr><td colspan=2>(none)</td></tr>"
    items = sorted(mapping.items(), key=lambda kv: kv[1], reverse=True)
    return "".join(
        f"<tr><td>{html.escape(str(k))}</td><td class=usd>{_usd(v)}</td></tr>"
        for k, v in items
    )


def _tree_lines(node: TreeNode, prefix: str = "", is_last: bool = True, root: bool = True) -> list[str]:
    span = node.span
    kind = span.kind.value
    label = span.name or span.model or kind
    text = f"{kind}: {label}" if kind != "trace" else f"trace {label if span.name else ''}".rstrip()
    toks = span.usage.total_tokens if span.usage else 0
    line_prefix = "" if root else f"{prefix}{'└─ ' if is_last else '├─ '}"
    lines = [f"{html.escape(line_prefix + text):<58}{toks:>10,} tok  {_usd(node.subtree_usd)}"]
    child_prefix = "" if root else prefix + ("   " if is_last else "│  ")
    for i, child in enumerate(node.children):
        lines += _tree_lines(child, child_prefix, i == len(node.children) - 1, root=False)
    return lines


def render_html(summary: Summary, traces: list[TraceReport], *, source: str) -> str:
    waste_html = "".join(
        f'<div class=flag>[{w.severity}] <strong>{html.escape(w.code)}</strong>'
        f'{f" ~{_usd(w.est_usd)}" if w.est_usd else ""} — {html.escape(w.message)}</div>'
        for w in summary.waste
    ) or "<p class=sub>(no waste flags)</p>"

    recon_html = ""
    if summary.reconciliation:
        rows = "".join(
            f"<tr><td>{html.escape(r.provider)}</td><td>{r.day}</td>"
            f"<td class=usd>{_usd(r.sdk_usd)}</td><td class=usd>{_usd(r.provider_usd)}</td>"
            f"<td class=usd>{r.drift_pct:.2f}%</td><td class=usd>{r.coverage_pct:.0f}%</td></tr>"
            for r in summary.reconciliation
        )
        worst = max(summary.reconciliation, key=lambda r: r.drift_pct)
        verdict = (
            f'<span class=ok>PASS ✅ (worst day {worst.drift_pct:.2f}%)</span>'
            if worst.drift_pct <= 2.0
            else f'<span class=warn>CHECK ⚠ (worst day {worst.drift_pct:.2f}%)</span>'
        )
        recon_html = (
            "<h2>Provider reconciliation (SDK vs billed)</h2>"
            "<table><tr><th>Provider</th><th>Day</th><th>SDK</th><th>Billed</th>"
            f"<th>Drift</th><th>Coverage</th></tr>{rows}</table>"
            f"<p class=sub>2% gate: {verdict}</p>"
        )

    unpriced_html = ""
    if summary.unpriced:
        items = ", ".join(f"{html.escape(m)} ({n})" for m, n in summary.unpriced.items())
        unpriced_html = (
            f'<div class=flag>⚠ <strong>Unpriced spans</strong> (recorded $0 — total is an '
            f"undercount): {items}</div>"
        )

    trees = "".join(
        f"<details><summary>{tr.trace_id[:12]} — "
        f"{html.escape(tr.feature_id or '(no feature)')} · {tr.n_spans} spans · "
        f"<strong>{_usd(tr.total_usd)}</strong></summary><pre>"
        + "\n".join(line for root in tr.roots for line in _tree_lines(root))
        + "</pre></details>"
        for tr in traces
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StepCost report</title><style>{_CSS}</style></head><body>
<h1>Step<span>Cost</span> report</h1>
<p class=sub>{html.escape(source)} · price-graph over {summary.n_traces} traces / {summary.n_spans} spans</p>
<div class=kpis>
  <div class=kpi><div class=v>{_usd(summary.total_usd)}</div><div class=l>total</div></div>
  <div class=kpi><div class=v>{summary.n_traces}</div><div class=l>traces</div></div>
  <div class=kpi><div class=v>{summary.n_spans}</div><div class=l>spans</div></div>
</div>
<h2>By feature</h2><table>{_rows(summary.by_feature)}</table>
<h2>By customer</h2><table>{_rows(summary.by_customer)}</table>
<h2>By kind</h2><table>{_rows(summary.by_kind)}</table>
<h2>Waste signals</h2>{waste_html}
{unpriced_html}
{recon_html}
<h2>Most expensive traces</h2>{trees or "<p class=sub>(none)</p>"}
<footer>Generated by <a href="https://stepcost.com">StepCost</a> — metadata only,
never prompt/response content.</footer>
</body></html>"""
