"""M4 — live-dollar reconciliation against a real Anthropic invoice.

Fires a handful of realistic Claude Haiku calls (cache on/off, varied input
sizes), captures StepCost's computed cost per call, and prints the total so you
can diff it against the actual charge in the Anthropic Console.

SAFE BY DEFAULT: without `--live` this prints the plan and exits WITHOUT calling
the API (no spend). Add `--live` to actually bill (~a few cents of Haiku).

The gate is a two-step flow — the live run's computed total is persisted, so
checking the gate later does NOT re-bill (re-running --live would also poison
the comparison: the second run's cache-write call becomes a cache *hit* within
the 5-minute TTL, and the Console total would include both runs):

    python scripts/haiku_reconcile.py                  # dry run, no spend
    python scripts/haiku_reconcile.py --live           # 1) real calls, persists total
    #   ...wait for Anthropic Console → Usage to show this run's charge...
    python scripts/haiku_reconcile.py --billed 0.0123  # 2) gate vs persisted total

Requires: `pip install anthropic` and ANTHROPIC_API_KEY (or an `ant auth login`
profile). Model: claude-haiku-4-5 (the current Haiku; 3.5-haiku is retired).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from stepcost import StepCost, Provider, llm_call
from stepcost.pricing import default_price_table
from stepcost.reconcile_gate import passes_reconciliation_gate, reconciliation_error_pct

MODEL = "claude-haiku-4-5"
RUN_FILE = Path.home() / ".stepcost" / "m4_last_run.json"

# A large static prefix so prompt caching actually engages (Haiku 4.5 needs a
# ~4096-token cacheable prefix). Repeated to comfortably clear that floor.
_CACHE_PREFIX = (
    "You are a meticulous financial-operations assistant. Follow every "
    "instruction precisely and answer tersely. " * 400
)
_SMALL_SYSTEM = "You are a terse assistant."

# (label, system_kind, user_prompt) — mix of small/large uncached + cache write/read.
CALLS = [
    ("small_no_cache", "small", "Reply with exactly the word: ok"),
    ("large_no_cache", "large", "Summarize in one sentence what a FinOps tool does."),
    ("cache_write", "cached", "Reply with exactly the word: one"),
    ("cache_read", "cached", "Reply with exactly the word: two"),
]


def _plan() -> None:
    print(f"Reconciliation plan — model {MODEL}")
    print(f"  price table version: {default_price_table().version}")
    pricing = default_price_table().models.get(MODEL)
    if pricing is None:
        print(f"  !! {MODEL} not in price table — add it before running --live")
        return
    print(
        f"  rates $/1M: in={pricing.input_per_1m} out={pricing.output_per_1m} "
        f"cache_read={pricing.cached_input_per_1m} cache_write={pricing.cache_write_per_1m}"
    )
    for label, system_kind, prompt in CALLS:
        print(f"  - {label:16} system={system_kind:6}  prompt={prompt[:40]!r}")
    print("\nDry run — no API calls made. Re-run with --live to bill (~a few cents).")


def _check_gate(computed: Decimal, billed: Decimal, *, tolerance: Decimal = Decimal("0.02")) -> bool:
    return passes_reconciliation_gate(computed, billed, tolerance=tolerance)


def _print_gate(computed: Decimal, billed: Decimal) -> bool:
    error = reconciliation_error_pct(computed, billed)
    pct = float(error * 100)
    ok = _check_gate(computed, billed)
    print("\n=== M4 reconciliation gate ===")
    print(f"  StepCost total:  ${computed:.6f}")
    print(f"  Anthropic billed: ${billed:.6f}")
    print(f"  Error:            {pct:.2f}%")
    print(f"  Gate (≤2%):       {'PASS ✅' if ok else 'FAIL ❌'}")
    if ok:
        print("\nRecord in docs/M4_RESULTS.md and mark BUILD.md M4 complete.")
    return ok


def _save_run(total: Decimal, rows: list[tuple[str, Decimal, str]]) -> None:
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUN_FILE.write_text(
        json.dumps(
            {
                "model": MODEL,
                "price_table_version": default_price_table().version,
                "computed_total_usd": str(total),
                "calls": [
                    {"label": label, "cost_usd": str(cost), "usage": usage}
                    for label, cost, usage in rows
                ],
                "ran_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )


def _load_run() -> dict | None:
    if not RUN_FILE.exists():
        return None
    return json.loads(RUN_FILE.read_text())


def check_gate_against_saved_run(billed: Decimal) -> int:
    run = _load_run()
    if run is None:
        print(
            f"error: no persisted live run at {RUN_FILE}.\n"
            "Run `python scripts/haiku_reconcile.py --live` first, wait for the "
            "charge to appear in the Anthropic Console, then re-run with --billed.",
            file=sys.stderr,
        )
        return 2
    computed = Decimal(run["computed_total_usd"])
    print(f"Comparing against persisted run from {run['ran_at']} (model {run['model']}).")
    return 0 if _print_gate(computed, billed) else 2


def run_live(rounds: int = 1) -> int:
    try:
        import anthropic
    except ImportError:
        print("error: `pip install anthropic` first (not a stepcost dependency).", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY or ant auth profile
    cc = StepCost(project="stepcost", environment="reconcile", sink="stdout", default_feature="m4")

    total = Decimal("0")
    rows: list[tuple[str, Decimal, str]] = []

    with cc.trace(feature_id="haiku_reconcile") as trace:
        for i in range(rounds):
            for label, system_kind, prompt in CALLS:
                if system_kind == "cached":
                    system = [
                        {
                            "type": "text",
                            "text": _CACHE_PREFIX,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ]
                elif system_kind == "large":
                    system = _CACHE_PREFIX
                else:
                    system = _SMALL_SYSTEM
                with llm_call(model=MODEL, provider=Provider.ANTHROPIC) as call:
                    resp = client.messages.create(
                        model=MODEL,
                        max_tokens=16,
                        system=system,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    call.record(resp, provider="anthropic")
                span_cost = call.span.cost.total_usd if call.span.cost else Decimal("0")
                u = resp.usage
                usage_str = (
                    f"in={u.input_tokens} out={u.output_tokens} "
                    f"cw={getattr(u, 'cache_creation_input_tokens', 0) or 0} "
                    f"cr={getattr(u, 'cache_read_input_tokens', 0) or 0}"
                )
                rows.append((label, span_cost, usage_str))
                total += span_cost
            if rounds > 1 and (i + 1) % 10 == 0:
                print(f"  ...round {i + 1}/{rounds}, running total ${total:.4f}", flush=True)

    cc.flush()

    print("\n=== StepCost computed cost per call (aggregated by label) ===")
    agg: dict[str, tuple[int, Decimal]] = {}
    for label, cost, _u in rows:
        n, c = agg.get(label, (0, Decimal("0")))
        agg[label] = (n + 1, c + cost)
    for label, (n, cost) in agg.items():
        print(f"  {label:16} x{n:<4} ${cost:.6f}")
    print(f"\nStepCost trace total: ${total:.6f}   (trace.total_usd=${trace.total_usd:.6f})")
    _save_run(total, rows)
    print(f"\nRun persisted to {RUN_FILE}.")
    print(
        f"Next: open Anthropic Console → Usage, filter to model {MODEL} and this "
        "run's window, then check the gate WITHOUT re-billing:\n"
        "  python scripts/haiku_reconcile.py --billed <console_total_usd>"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live", action="store_true", help="actually call the API (bills real $)")
    ap.add_argument(
        "--rounds", type=int, default=1,
        help="repeat the call set N times (size the run so cents-rounding can't blur the gate)",
    )
    ap.add_argument(
        "--billed",
        type=Decimal,
        default=None,
        help="actual $ from Anthropic Console — gates against the persisted --live run",
    )
    args = ap.parse_args(argv)
    if args.billed is not None and args.billed <= 0:
        print("error: --billed must be a positive dollar amount.", file=sys.stderr)
        return 2
    if args.live and args.billed is not None:
        print(
            "error: --live and --billed are separate steps. A fresh --live run can't be "
            "compared to a Console figure from before it ran (the re-run's cache-write "
            "call becomes a cache hit, and the Console total would include both runs).\n"
            "Run --live alone, wait for the Console charge, then run --billed alone.",
            file=sys.stderr,
        )
        return 2
    if args.billed is not None:
        return check_gate_against_saved_run(args.billed)
    if not args.live:
        _plan()
        return 0
    return run_live(rounds=args.rounds)


if __name__ == "__main__":
    raise SystemExit(main())
