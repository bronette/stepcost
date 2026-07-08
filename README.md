# StepCost

**FinOps for LLM agents** — unit economics and cost accountability in the developer workflow.

> See what every agent step costs — and block the PR that doubles your token bill.

## Why StepCost

| Us | Them (Langfuse, Helicone, LiteLLM) |
|----|-------------------------------------|
| **Agent economics graph** — typed cost nodes (LLM, tool, RAG, cache, retry) | Flat traces; cost as a dashboard field |
| **Cost-as-Code** — GitHub PR comments + CI budgets (planned) | Dashboards engineers ignore |
| **Finance-grade ledger** — versioned price table, point-in-time $ | Debug-time estimates |

Invoice-grade by design: cache reads (0.1x), 5-minute cache writes (1.25x),
1-hour cache writes (2x), and reasoning tokens are priced the way your provider
bills them — for OpenAI (Chat Completions, Responses API, embeddings) and
Anthropic. Unrecognized usage shapes raise instead of silently recording $0,
and models missing from the price table are flagged loudly in every report.

## Quick start (5 minutes, $0, no API key)

```bash
cd sdk/python
pip install -e .
python examples/quickstart.py                 # instruments a tiny agent, writes a cost graph
stepcost report ~/.stepcost/quickstart.db     # render the per-step cost tree
```

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

## LangChain / LangGraph (zero lines per call site)

```bash
pip install "stepcost[langchain]"
```

```python
from stepcost import StepCost
from stepcost.integrations.langchain import StepCostCallbackHandler

cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")
handler = StepCostCallbackHandler(cc)

with cc.trace(feature_id="support-bot", customer_id="acme") as trace:
    agent.invoke(inputs, config={"callbacks": [handler]})

print(trace.total_usd, trace.by_step)
```

Every chain/graph node becomes a priced `agent_step`, every model call an
`llm_generation` (cache reads/writes and reasoning tokens unfolded from
LangChain's aggregated `usage_metadata` — no double-counting), every tool call
a `tool_call` — parented by LangChain's own run tree, so parallel branches
attribute correctly. LCEL plumbing (`RunnableSequence` etc.) is filtered out of
the tree.

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
