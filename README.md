# StepCost

**FinOps for LLM agents** — unit economics and cost accountability in the developer workflow.

> See what every agent step costs — and block the PR that doubles your token bill.

## Why StepCost

| Us | Them (Langfuse, Helicone, LiteLLM) |
|----|-------------------------------------|
| **Agent economics graph** — typed cost nodes (LLM, tool, RAG, cache, retry) | Flat traces; cost as a dashboard field |
| **Cost-as-Code** — GitHub PR comments + CI budgets (beta) | Dashboards engineers ignore |
| **Finance-grade ledger** — versioned price table, point-in-time $ | Debug-time estimates |

Invoice-grade by design: cache reads (0.1x), 5-minute cache writes (1.25x),
1-hour cache writes (2x), and reasoning tokens are priced the way your provider
bills them — for OpenAI (Chat Completions, Responses API, embeddings) and
Anthropic. Unrecognized usage shapes raise instead of silently recording $0,
and models missing from the price table are flagged loudly in every report.
**Verified against a real Anthropic invoice: 0.0016% error** on a live
reconciliation run (July 2026), gated by `stepcost sync` against the org
cost API.

## Quick start (5 minutes, $0, no API key)

```bash
pip install stepcost
curl -sO https://raw.githubusercontent.com/bronette/stepcost/main/sdk/python/examples/quickstart.py
python quickstart.py                          # instruments a tiny agent, writes a cost graph
stepcost report ~/.stepcost/quickstart.db     # render the per-step cost tree
```

(From a clone: `cd sdk/python && pip install -e . && python examples/quickstart.py`)

That runs a 3-step agent with simulated token usage — no API calls, no spend —
and renders a priced cost tree with by-step / by-customer rollups and waste flags.

## Instrument a real agent (5 lines)

```python
from stepcost import StepCost, agent_step, llm_call

cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")

with cc.trace(feature_id="support-bot", customer_id="acme") as trace:
    with agent_step("answer"), llm_call(model="gpt-4o-mini", provider="openai") as call:
        response = openai_client.chat.completions.create(...)
        call.record(response)  # invoice-safe token + $ extraction from the response

print(trace.total_usd, trace.by_step)
```

Spans persist at trace and process exit (no manual flush needed), sink failures
re-queue instead of dropping, and concurrent async agents keep correct
parent/child cost attribution. For providers without an extractor, call
`record_usage(input_tokens=..., output_tokens=...)`.

## Integrations (zero lines per call site)

```python
# LangChain / LangGraph — one callback handler        pip install "stepcost[langchain]"
from stepcost.integrations.langchain import StepCostCallbackHandler
agent.invoke(inputs, config={"callbacks": [StepCostCallbackHandler(cc)]})

# LiteLLM — 100+ providers, any language via the proxy
from stepcost.integrations.litellm import StepCostLiteLLMLogger
litellm.success_callback = [StepCostLiteLLMLogger(cc).log_success_event]

# Direct SDK clients — wrap once, every call priced
from stepcost.integrations.openai import instrument_openai
from stepcost.integrations.anthropic import instrument_anthropic
client = instrument_openai(OpenAI(), cc)
claude = instrument_anthropic(anthropic.Anthropic(), cc)
```

All four produce the same typed cost graph and compose inside one trace. Full
details and the manual API: [`docs/INTEGRATIONS.md`](docs/INTEGRATIONS.md).

## Cost-as-Code (beta): the PR comment that prices your agent changes

```bash
stepcost diff base.db head.db --budget .stepcost.toml --markdown   # exit 3 over budget
stepcost report ~/.stepcost/my-app.db --html report.html           # local dashboard file
```

Run your instrumented agent suite on the base branch and the PR branch, diff
the two databases, and post the delta as a PR comment — the repo doubles as a
GitHub Action (`uses: bronette/stepcost@main`). Budgets live in
`.stepcost.toml` (`max_total_usd`, `max_trace_usd`, `max_increase_pct`).
Guide: [stepcost.com/docs/cost-as-code.html](https://stepcost.com/docs/cost-as-code.html).

## Two-sided ledger: reconcile against what your provider actually bills

```bash
export ANTHROPIC_ADMIN_KEY=sk-ant-admin01-...   # Console → Settings → Organization → Admin keys
stepcost sync anthropic ~/.stepcost/my-app.db --days 7
stepcost sync openai    ~/.stepcost/my-app.db --days 7   # OPENAI_ADMIN_KEY
stepcost report ~/.stepcost/my-app.db
```

`sync` pulls the org Cost APIs (what the provider says it will bill, per day)
into the same SQLite file, and the report gains a reconciliation section:

```
Provider reconciliation (SDK-observed vs provider-billed):
  anthropic  2026-07-08   SDK $0.0190   billed $0.0190   drift 0.2%   coverage 100%
  2% gate (worst day): 0.23% — PASS ✅
```

**Drift** audits StepCost's pricing against the invoice, continuously.
**Coverage** catches what no SDK can see alone: spend from call sites you never
instrumented. The provider knows *what you'll pay*; the SDK knows *which step,
feature, and customer caused it* — each side audits the other's blind spot.

## Privacy

**Metadata-only by default.** Spans capture token counts, model, and your
business dimensions (feature/customer/step) — **never prompt or response
content** (`PayloadCapture.NONE`). The SDK works fully offline; the SQLite sink
stays on your machine.

## Docs

- [`SPEC.md`](SPEC.md) — product + technical design
- [`sdk/python/README.md`](sdk/python/README.md) — SDK reference, report CLI, development
- [stepcost.com](https://stepcost.com) — landing page

## Design partners

Shipping agents and burning API tokens? We're looking for teams to instrument
one agent for 2 weeks — free. Email
[kevin.brunette@gmail.com](mailto:kevin.brunette@gmail.com?subject=StepCost%20design%20partner).

## License

MIT — see [`LICENSE`](LICENSE).
