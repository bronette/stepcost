# Integrations

Four ways to get spans flowing, from zero-code to fully manual. All of them
produce the same typed cost graph (`trace → agent_step → llm/tool/retrieval/
embedding`) with invoice-safe token extraction — pick per call site; they
compose freely inside one trace.

| Integration | Code per call site | Covers |
|---|---|---|
| [LangChain / LangGraph](#langchain--langgraph) | 0 lines (one callback) | chains, graphs, tools, retrievers, any LC model |
| [LiteLLM](#litellm) | 0 lines (one callback) | 100+ providers routed through LiteLLM |
| [OpenAI / Anthropic client wrappers](#client-wrappers) | 0 lines (wrap once) | direct SDK usage |
| [Manual context managers](#manual) | 2 lines | anything else |

## LangChain / LangGraph

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
```

Every chain/graph node becomes an `agent_step`, model calls become priced
`llm_generation` spans (cache reads/writes and reasoning tokens unfolded from
LangChain's aggregated `usage_metadata` — no double-counting), tools and
retrievers included. Parenting follows LangChain's own run tree, so parallel
branches attribute correctly; LCEL plumbing (`RunnableSequence` etc.) is
filtered out.

## LiteLLM

No extra dependency — LiteLLM accepts plain callables:

```python
import litellm
from stepcost.integrations.litellm import StepCostLiteLLMLogger

logger = StepCostLiteLLMLogger(cc)
litellm.success_callback = [logger.log_success_event]
litellm.failure_callback = [logger.log_failure_event]

with cc.trace(feature_id="chat") as trace:
    litellm.completion(model="claude-haiku-4-5", messages=[...])
```

Handles LiteLLM's normalized usage (OpenAI-style cached-subset subtraction,
Anthropic cache read/write counters) and records call duration. Because the
LiteLLM proxy sees traffic from every language, this is also the cheapest way
to cover non-Python services.

## Client wrappers

Wrap the official SDK clients once; every call is priced automatically:

```python
from stepcost.integrations.openai import instrument_openai
from stepcost.integrations.anthropic import instrument_anthropic

openai_client = instrument_openai(OpenAI(), cc)          # chat, responses, embeddings
claude_client = instrument_anthropic(anthropic.Anthropic(), cc)  # messages

with cc.trace(feature_id="chat") as trace:
    openai_client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    claude_client.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=[...])
```

Spans nest under whatever `agent_step` is open at the call site. Provider
errors re-raise untouched but still close the span with an `error` tag.
Streaming responses (no usage object) record `usage=unavailable` instead of
guessing.

## Manual

For anything else — custom providers, tool calls, retrieval steps:

```python
from stepcost import agent_step, llm_call

with cc.trace(feature_id="pipeline") as trace:
    with agent_step("summarize"):
        with llm_call(model="my-model", provider="other") as call:
            resp = my_client.generate(...)
            call.record_usage(input_tokens=resp.in_toks, output_tokens=resp.out_toks)
```

Unknown models warn loudly and show as "unpriced spans" in `stepcost report` —
add them with `register_custom_model()` or a custom price table.
