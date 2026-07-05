"""StepCost 5-minute quickstart — runs with NO API key and $0 of real spend.

It instruments a tiny 3-step agent, records realistic token usage, and writes a
priced cost graph to a local SQLite file. Then render it:

    python examples/quickstart.py
    stepcost report ~/.stepcost/quickstart.db

You'll see a per-step cost tree, by-step / by-kind rollups, and any waste flags —
the same output you'd get on a real agent, without spending anything.

To instrument a *real* agent, replace the `record_usage(...)` calls with
`call.record(response)` right after your OpenAI/Anthropic API call — StepCost
extracts invoice-safe token counts from the response object.
"""

from __future__ import annotations

from pathlib import Path

from stepcost import SpanKind, StepCost, agent_step, llm_call

DB = Path.home() / ".stepcost" / "quickstart.db"
DB.parent.mkdir(parents=True, exist_ok=True)

cc = StepCost(
    project="quickstart",
    environment="dev",
    sink=f"sqlite:///{DB}",
    default_feature="support-bot",
)

with cc.trace(customer_id="acme", feature_id="support-bot") as trace:
    # 1) Plan: one LLM call.
    with agent_step("plan"):
        with llm_call(model="gpt-4o-mini", provider="openai") as call:
            # In real code: resp = openai.chat.completions.create(...); call.record(resp)
            call.record_usage(input_tokens=1_200, output_tokens=180)

    # 2) Retrieve: an embedding to search a vector store.
    with agent_step("retrieve"):
        with cc.span(
            kind=SpanKind.EMBEDDING,
            model="text-embedding-3-small",
            provider="openai",
        ) as emb:
            emb.set_metadata(tool="pinecone")
            emb.record_usage(embedding_tokens=8_000)

    # 3) Execute: a bigger generation on a frontier model.
    with agent_step("execute"):
        with llm_call(model="gpt-4o", provider="openai") as call:
            call.record_usage(input_tokens=3_500, output_tokens=600)

cc.flush()

print(f"trace total: ${trace.total_usd:.4f}")
print(f"by step:     {trace.by_step}")
print(f"by kind:     {trace.by_kind}")
print(f"\nWrote cost graph to {DB}")
print(f"Now render it:\n    stepcost report {DB}")
