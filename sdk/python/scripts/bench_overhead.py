"""Span-emit overhead micro-benchmark (StepCost M3 gate: p99 < 5ms).

Measures the wall-clock cost the SDK adds per instrumented LLM call: opening an
`llm_call` span, recording usage (which prices it), and finishing (index + emit
to a batched sink). This is the overhead an app pays on top of its real work.

Run:  python scripts/bench_overhead.py [N]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from stepcost import StepCost, agent_step, llm_call


def _percentile(sorted_us: list[float], p: float) -> float:
    if not sorted_us:
        return 0.0
    k = max(0, min(len(sorted_us) - 1, round((p / 100.0) * (len(sorted_us) - 1))))
    return sorted_us[k]


def run(n: int) -> int:
    with TemporaryDirectory() as td:
        db = Path(td) / "bench.db"
        cc = StepCost(project="bench", sink=f"sqlite:///{db}", default_feature="bench")
        samples_us: list[float] = []

        with cc.trace(customer_id="bench", feature_id="bench"):
            # warm up (import/JIT/first-span allocations) — excluded from stats
            for _ in range(200):
                with agent_step("warm"), llm_call(model="gpt-4o-mini") as c:
                    c.record_usage(input_tokens=1000, output_tokens=200)

            for _ in range(n):
                t0 = time.perf_counter()
                with agent_step("step"), llm_call(model="gpt-4o-mini") as c:
                    c.record_usage(input_tokens=1000, output_tokens=200)
                samples_us.append((time.perf_counter() - t0) * 1_000_000)

        cc.flush()

    samples_us.sort()
    p50 = _percentile(samples_us, 50)
    p99 = _percentile(samples_us, 99)
    p999 = _percentile(samples_us, 99.9)
    mean = sum(samples_us) / len(samples_us)

    print(f"samples:        {len(samples_us):,} (agent_step + llm_call + record_usage)")
    print(f"mean:           {mean:8.1f} us")
    print(f"p50:            {p50:8.1f} us")
    print(f"p99:            {p99:8.1f} us  ({p99 / 1000:.3f} ms)")
    print(f"p99.9:          {p999:8.1f} us  ({p999 / 1000:.3f} ms)")
    print(f"max:            {samples_us[-1]:8.1f} us")

    ok = p99 < 5000
    print(f"\nGATE p99 < 5ms: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    raise SystemExit(run(n))
