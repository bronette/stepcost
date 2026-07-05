# StepCost — Product & Technical Specification

*Version 0.1 · 2026-06-29 · Status: design + MVP scaffold*

---

## 1. One-liner

**StepCost is the FinOps layer for LLM agents — unit economics and cost accountability in the developer workflow, not another trace viewer.**

---

## 2. Value edge (why we win)

### The problem nobody owns end-to-end

| Player | What they do well | Where they stop |
|--------|-------------------|-----------------|
| **Langfuse / Helicone** | Traces, token logs, dashboards | Cost is a *field on a span*; feature/customer = loose `metadata`; no PR/CI workflow |
| **LiteLLM / Portkey** | Proxy, routing, budgets at gateway | No product unit economics; no agent-step graph; no finance audit trail |
| **CloudZero / Finout** | Cloud $ allocation, "TokenOps" marketing | Bolted-on AI; no agent primitives; top-down enterprise sales |
| **Infracost** | `$` diff on infra PRs | Zero LLM/agent coverage |

Buyers want **visibility, allocation, ROI** first. Observability tools show the bill; acting on it (PR/CI cost gates) is the follow-on step we're validating with design partners.

### Our edge — three pillars (defensible together)

#### Pillar A: **Agent Economics Graph** (not flat traces)

Observability tools record spans. We record a **typed cost graph**:

```
trace (agent run)
├── agent_step: planner
│   ├── llm_generation  ($0.042)
│   └── tool_call: search  ($0.008 embedding + $0.001 retrieval)
├── agent_step: executor
│   ├── llm_generation  ($0.031)
│   └── cache_read  ($0.002 saved)
└── rollup → $0.084 / run
```

**First-class dimensions** (required schema, not optional metadata):

- `customer_id`, `feature_id`, `team_id`, `environment`
- `prompt_version`, `model`, `provider`
- Node type: `llm_generation | tool_call | retrieval | embedding | cache_read | cache_write | retry | agent_step`

**Why it matters:** Langfuse can tag `metadata.feature = "copilot"` — but finance can't audit rollups, compare agent *steps*, or answer "which tool loop is burning margin on customer Acme?" without custom ETL. We make **unit economics the data model**, not a dashboard filter.

#### Pillar B: **Cost-as-Code** (distribution + *optional* action)

> ⚠️ **Hypothesis, not verified:** engineers will block/warn on PRs for token cost (E8 unverified). Ship **cost-on-PR as comment first**; CI gate only if design partners ask for it.

Meet engineers where spend is introduced:

| Surface | Behavior |
|---------|----------|
| **GitHub App** | Comment on every PR touching prompts/agents: token delta, projected $/mo, model mix shift |
| **CI check** | Fail or warn when PR exceeds token budget or $ regression threshold |
| **Policy file** | `.stepcost.yml` — budgets per feature/env, allowed models |

**Why it matters:** No major LLM observability player ships a native **cost-on-PR** product. Infracost proved `$ in the PR` wins bottom-up adoption. We are **Infracost for LLM agents**.

#### Pillar C: **Finance-grade cost ledger** (trust the number)

- **Versioned price table** — costs computed at ingest time using prices *as of that timestamp*
- **Provider-safe extraction** — OpenAI: `prompt_tokens` includes cached (split, don't double-count); o-series: reasoning split from `completion_tokens`. Anthropic: separate cache read/write lines.
- **Reconciliation tests** — extracted $ must match manual invoice line items within 2% on fixture corpus; expand with real invoice CSV before marketing "finance-grade"
- **Margin hooks** — join `customer_id` + `revenue_cents` for gross-margin per AI feature (Phase 2 dashboard)

**Why it matters:** Eng teams distrust "dashboard estimates." Finance needs point-in-time auditability. Observability vendors optimize for debugging, not SOX-friendly allocation.

### Positioning sentence (GTM)

> **"See what every agent step costs — and block the PR that doubles your token bill."**

### What we explicitly do NOT claim as edge

- Generic trace UI (commoditized)
- "Save 40–85%" without measured cohort data
- Multi-cloud FinOps
- K8s/GPU optimization
- Being the LLM proxy (LiteLLM owns that primitive)

---

## 3. Ideal customer profile (ICP)

### Phase 1 design partners

- AI-native product companies
- **$20K–$200K/mo** API token spend (pain is acute)
- Shipping **agents** (multi-step, tools, RAG) — not single-shot chat
- Eng-led, GitHub-centric
- No dedicated FinOps team yet

### Phase 2 paid

- Same profile + first **AI SaaS** teams needing cost-per-customer for pricing decisions

### Phase 3 enterprise

- Regulated / data-sovereignty buyers needing **self-hosted control plane**
- Platform teams standardizing LLM spend across 50+ services

---

## 4. Product scope by phase

### v0.1 — SDK + local sink (weeks 1–3)

| In | Out |
|----|-----|
| Python SDK with typed spans | Hosted dashboard |
| OpenAI + Anthropic usage extractors (`call.record(response)`) | TypeScript SDK |
| Versioned price table + reconciliation tests | GitHub App (spec only) |
| In-memory rollups (`total_usd`, `by_kind`, `by_step`) | SSO, Helm |
| Local SQLite / stdout sink | Gateway/proxy mode |
| Contextvar DX (`agent_step("plan")` without threading `cc`) | Billing/invoicing |
| Agent-step parent/child context | |
| Dogfood on Oracle | |

### v0.2 — GitHub App + cloud ingest (weeks 4–6)

- GitHub App: cost comment on prompt/agent file changes
- Hosted ingest API (multi-tenant)
- Minimal web UI: trace cost drill-down + feature rollup

### v0.3 — First revenue (months 3–4)

- Dashboard: cost per feature / customer / agent-step
- Anomaly alerts (runaway loop detection)
- `.stepcost.yml` CI budget check (GitHub Action)

### v1.0 — Enterprise tier (months 6–12)

- Self-hosted control plane (same images as cloud + license key)
- SSO/SAML, RBAC, audit log, retention policies
- SOC 2 Type I
- Optional: prompt-optimization PR bot (closed loop)

---

## 5. System architecture

### Hybrid deployment (design constraint from day 1)

```
┌──────────────────────────────────────────────────────────────────┐
│                        Application / CI                          │
│  stepcost SDK  ──►  span batcher  ──►  sink (pluggable)   │
└───────────────────────────────┬──────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
    ┌──────▼──────┐     ┌───────▼───────┐    ┌──────▼──────┐
    │ Local sink   │     │ Cloud ingest  │    │ Self-host   │
    │ SQLite/stdout│     │ (SaaS default)│    │ ingest API  │
    └──────────────┘     └───────┬───────┘    └──────┬──────┘
                                 │                    │
                          ┌──────▼────────────────────▼──────┐
                          │     Control plane (Postgres)    │
                          │  rollups · alerts · dashboards  │
                          └─────────────────────────────────┘
```

**Rules:**

1. SDK never requires cloud — works fully offline.
2. Cloud and self-host run **identical control-plane code**.
3. Enterprise features = license key on self-host (Langfuse pattern).
4. Sensitive payloads (prompts, tool I/O) are **opt-in**; cost spans default to **metadata-only**.

---

## 6. Core data model

### 6.1 Span (atomic unit)

Every instrumented operation emits one `Span`:

```yaml
span_id: uuid
trace_id: uuid
parent_span_id: uuid | null

# Taxonomy
kind: llm_generation | tool_call | retrieval | embedding |
      cache_read | cache_write | retry | agent_step | trace

# Identity (first-class allocation)
org_id: string          # tenant
project_id: string
customer_id: string | null
feature_id: string | null
team_id: string | null
environment: string     # prod | staging | dev

# LLM specifics (when kind = llm_generation)
provider: openai | anthropic | bedrock | vertex | ollama | other
model: string
prompt_version: string | null

# Usage
usage:
  input_tokens: int
  output_tokens: int
  cached_input_tokens: int      # prompt cache hits
  reasoning_tokens: int         # o1-style hidden reasoning
  cache_creation_tokens: int    # Anthropic 5m-TTL cache write (1.25x input)
  cache_creation_1h_tokens: int # Anthropic 1h-TTL cache write (2x input)
  embedding_tokens: int         # embedding calls (priced via embedding_per_1m)

# Cost (computed at ingest from versioned price table)
cost:
  input_usd: decimal
  output_usd: decimal
  cached_input_usd: decimal
  reasoning_usd: decimal
  total_usd: decimal
  price_table_version: string   # e.g. "2026-07-05"

# Timing
started_at: iso8601
duration_ms: int

# Optional debug (off by default in prod)
payload:
  capture: none | hash | full
  prompt_hash: sha256 | null
  response_hash: sha256 | null

# Correlation
session_id: string | null
user_id: string | null
tags: dict[str, str]
```

### 6.2 Trace (logical agent run)

A trace is the root `span` (`kind=trace`) plus all descendants. Rollups:

- `trace.total_usd` = sum descendant `cost.total_usd`
- `trace.by_kind` = breakdown by span kind
- `trace.by_step` = breakdown by `agent_step` nodes
- `trace.waste_signals` = retries, duplicate retrievals, cache misses (Phase 2)

### 6.3 Price table (versioned)

```yaml
version: "2026-07-05"
models:
  gpt-4o:
    provider: openai
    input_per_1m: 2.50
    output_per_1m: 10.00
    cached_input_per_1m: 1.25
  claude-sonnet-4-20250514:
    provider: anthropic
    input_per_1m: 3.00
    output_per_1m: 15.00
    cached_input_per_1m: 0.30    # cache-read rate (field name matches code/price_table.json)
    cache_write_per_1m: 3.75     # 5m-TTL cache write (1.25x input)
    cache_write_1h_per_1m: 6.00  # 1h-TTL cache write (2x input)
```

- Shipped as JSON in SDK repo; overridable for custom/self-hosted models
- **Immutable versions** — new provider prices = new version, never retroactive mutation
  (`ModelPricing` is frozen in code, so a cached table can't be mutated at runtime)
- Unknown model → span records $0 with `price_table_version = null`, a once-per-model
  runtime warning, and an "unpriced spans" section in `stepcost report`

### 6.4 Policy file (`.stepcost.yml`)

```yaml
version: 1
project: my-ai-product

budgets:
  - feature: support-copilot
    env: prod
    monthly_usd: 5000
    alert_at_pct: 80

  - feature: doc-summary
    env: prod
    per_trace_usd: 0.50

ci:
  on_pull_request:
    warn_if_monthly_projection_usd_delta_gt: 100
    fail_if_monthly_projection_usd_delta_gt: 500
    paths:
      - "prompts/**"
      - "**/agents/**"
      - "**/*prompt*.yaml"

models:
  allow:
    - gpt-4o-mini
    - claude-sonnet-4-20250514
  block:
    - gpt-4o  # except in prod — enforced in CI later
```

---

## 7. SDK design (Python v0.1)

### 7.1 Package layout

```
stepcost/
├── SPEC.md
├── README.md
├── sdk/
│   └── python/
│       ├── pyproject.toml
│       ├── stepcost/
│       │   ├── __init__.py
│       │   ├── client.py       # StepCost client
│       │   ├── context.py      # trace/step context managers
│       │   ├── models.py       # Pydantic span models
│       │   ├── pricing.py      # price table + cost calc
│       │   ├── wrappers/       # openai, anthropic (v0.2)
│       │   └── sinks/
│       │       ├── base.py
│       │       ├── stdout.py
│       │       ├── sqlite.py
│       │       └── http.py     # cloud ingest (v0.2)
│       └── tests/
└── github-app/                 # (v0.2)
```

### 7.2 Public API (Python)

```python
from stepcost import StepCost, SpanKind, agent_step, llm_call

cc = StepCost(
    project="oracle",
    environment="dev",
    sink="sqlite:///~/.stepcost/oracle.db",
    # sink="https://ingest.stepcost.dev",  # v0.2
    default_feature="debate",
)

# Context managers — agent economics graph
with cc.trace(customer_id="acme", feature_id="support-bot") as trace:
    with agent_step("plan"):
        with llm_call(model="gpt-4o-mini") as call:
            response = openai_client.chat.completions.create(...)
            call.record(response)  # auto token + $ extraction

    with agent_step("retrieve"):
        with cc.span(
            kind=SpanKind.EMBEDDING,
            model="text-embedding-3-small",  # model → priced embedding cost
            provider="openai",
        ) as s:
            s.record_usage(embedding_tokens=1200)
            s.set_metadata(tool="pinecone", chunks="5")

# Decorator form (v0.2 — NOT in v0.1; use the context managers above for now)
@cc.instrument(feature_id="summarize")
def generate_summary(user_id: str, doc: str) -> str:
    ...

# NOTE: embedding_tokens on retrieval/embedding spans shipped in v0.1.
```

### 7.3 Wrapper integrations (v0.2)

- `stepcost.integrations.openai` — drop-in patch or wrapper
- `stepcost.integrations.anthropic`
- `stepcost.integrations.litellm` — hook completion callback

Priority: **manual context API first** (no monkey-patching magic in v0.1).

### 7.4 Sink interface

```python
class SpanSink(Protocol):
    def emit(self, spans: list[Span]) -> None: ...
    def flush(self) -> None: ...
```

Default batching: flush every 100 spans or 5 seconds.

---

## 8. Cloud ingest API (v0.2 sketch)

```
POST /v1/spans/batch
Authorization: Bearer cc_live_...
Content-Type: application/json

{ "spans": [ ... ] }
```

- Idempotent on `span_id`
- Server recomputes cost if client omitted (uses server price table at `started_at`)
- Rate limit per project API key

---

## 9. GitHub App (v0.2 sketch)

### Trigger

PR opened/synchronized; diff touches configured paths (`prompts/**`, agent configs).

### Analysis pipeline

1. Checkout base + head
2. Static prompt diff → token estimate (tiktoken / model tokenizer)
3. If CI traces exist for both refs → use **measured** spans (preferred)
4. Compute Δ tokens, Δ projected monthly $ (uses repo traffic multiplier from config)
5. Post/update PR comment

### Comment template

```markdown
## StepCost — LLM cost impact

| Metric | Base | This PR | Δ |
|--------|------|---------|---|
| Est. tokens / request | 4,200 | 6,800 | +62% |
| Est. cost / request | $0.008 | $0.013 | +$0.005 |
| Projected monthly (at 100K req) | $800 | $1,300 | **+$500** |

**Files:** `prompts/support/system.md` (+1,400 tokens)

<details>
<summary>Agent step breakdown (measured from CI trace)</summary>
...
</details>

> Set budgets in `.stepcost.yml`. [Docs](...)
```

---

## 10. Security & privacy

| Concern | v0.1 approach |
|---------|---------------|
| Prompt content in spans | **Default: none.** Hash-only optional. |
| PII in tool outputs | Never captured unless `payload.capture=full` + explicit opt-in |
| API keys | SDK keys scoped to project; rotatable |
| Self-host | No outbound telemetry required; air-gap supported (v1) |
| Multi-tenancy | Row-level `org_id` isolation; no cross-tenant queries |

---

## 11. Success metrics

### Design partner gate (go/no-go for hosted product)

- [ ] SDK in **5+ production codebases**
- [ ] **1+ "holy shit" moment** per partner (agent loop cost revealed)
- [ ] GitHub App installed on **3+ repos**
- [ ] Partners say they'd pay **$1–4K/mo** for dashboard + alerts

### Technical gate

- [ ] Cost within **2% of provider invoice** on reconciled sample
- [ ] < **5ms p99 overhead** per span emit (batched)
- [ ] SDK works **fully offline**

---

## 12. Naming & repo

- **Product:** StepCost (working name)
- **Python package:** `stepcost`
- **Repo folder:** `~/code-projects/stepcost/`
- Public brand TBD before launch

---

## 13. Open decisions

| # | Question | Recommendation |
|---|----------|----------------|
| 1 | TypeScript SDK before or after GitHub App? | **After** — GH App can call Python CLI for v0.2 |
| 2 | Open-core vs fully OSS SDK? | **SDK fully OSS (MIT)**; hosted + enterprise license monetize |
| 3 | Store spans in ClickHouse or Postgres first? | **Postgres** until >100M spans/mo |
| 4 | Product name for market | Decide before public launch |

---

## 14. Immediate build order (this sprint)

1. ✅ This spec
2. Python `models.py` + `pricing.py` + `client.py` + SQLite sink
3. Dogfood against a real local agent workload
4. Unit tests: cost calc, span rollups, price table versioning
