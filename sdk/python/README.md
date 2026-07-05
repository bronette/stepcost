# stepcost (Python SDK)

FinOps for LLM agents — a typed, priced cost graph for every agent step. See
[`../../SPEC.md`](../../SPEC.md) for the product and technical design.

## Install

```bash
pip install -e .            # runtime only
pip install -e ".[dev]"     # + pytest, ruff
```

Requires Python ≥ 3.11. Runtime deps: `pydantic` (only).

## Quickstart (no API key, $0)

```bash
python examples/quickstart.py
stepcost report ~/.stepcost/quickstart.db
```

`examples/quickstart.py` instruments a 3-step agent (plan → retrieve → execute)
with simulated usage and writes a priced cost graph to SQLite. `stepcost report`
renders the per-step tree, by-step / by-kind / by-customer rollups, and waste
flags — no spend, no network.

## Instrument your agent

```python
from stepcost import StepCost, agent_step, llm_call

cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")

with cc.trace(feature_id="chat", customer_id="acme") as trace:
    with agent_step("answer"), llm_call(model="gpt-4o-mini", provider="openai") as call:
        resp = openai_client.chat.completions.create(...)
        call.record(resp)               # invoice-safe token + $ extraction

print(trace.total_usd, trace.by_step)   # also .by_kind
```

- `call.record(resp)` extracts usage from OpenAI (Chat Completions, Responses
  API, embeddings) **and** Anthropic responses — cached / reasoning tokens and
  5m- vs 1h-TTL cache writes split correctly, no double-count. Unrecognized
  usage shapes raise instead of silently recording $0.
- No extractor for your provider? Use `record_usage(input_tokens=…, output_tokens=…,
  embedding_tokens=…)`.
- Sinks: `sink="stdout"` or `sink="sqlite:///path.db"`.
- Safe by construction: spans persist at trace exit and process exit (no manual
  `flush()` needed), sink failures re-queue instead of dropping spans, and
  concurrent asyncio tasks under one trace parent their spans correctly.
- Models missing from the price table trigger a runtime warning and an
  "unpriced spans" section in `stepcost report` — never a silent $0.

## Report CLI

```bash
stepcost report <db>                 # account summary: by feature/customer/kind, top traces, waste
stepcost report <db> --trace <id>    # one trace's cost tree (per-node tokens + $)
stepcost report <db> --json          # machine-readable
```

## Privacy

Metadata-only by default (`PayloadCapture.NONE`): token counts, model, and your
business dimensions — never prompt/response content. Fully offline; SQLite stays
local.

## Develop

```bash
pytest        # 50 tests
ruff check stepcost tests scripts
python scripts/bench_overhead.py     # p99 span-emit overhead (<5ms gate)
```
