"""LangChain / LangGraph integration — zero-line instrumentation.

Usage::

    from stepcost import StepCost
    from stepcost.integrations.langchain import StepCostCallbackHandler

    cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")
    handler = StepCostCallbackHandler(cc)

    with cc.trace(feature_id="support-bot", customer_id="acme") as trace:
        agent.invoke(inputs, config={"callbacks": [handler]})

    print(trace.total_usd, trace.by_step)

Every chain/agent node becomes an ``agent_step`` span, every LLM call an
``llm_generation`` span (with invoice-safe token extraction), every tool call a
``tool_call`` span — parented by LangChain's own run tree, so concurrent
branches attribute correctly.

Requires ``langchain-core`` (``pip install "stepcost[langchain]"``).
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "StepCostCallbackHandler requires langchain-core. "
        'Install it with: pip install "stepcost[langchain]"'
    ) from exc

from stepcost.client import StepCost
from stepcost.context import ActiveSpan
from stepcost.models import Provider, SpanKind

# LCEL plumbing fires chain callbacks for every Runnable; these add tree noise
# without being meaningful "steps". Their run_ids pass through to the nearest
# meaningful ancestor instead of creating spans.
_PASSTHROUGH_PREFIXES = ("Runnable", "ChannelWrite", "ChannelRead", "_")
_PASSTHROUGH_NAMES = {"LangGraph", "__start__", "__end__", "should_continue"}


def _guess_provider(serialized: dict | None, params: dict) -> Provider:
    hay = " ".join(
        str(x).lower()
        for x in (
            (serialized or {}).get("id", []),
            params.get("_type", ""),
            params.get("model", ""),
            params.get("model_name", ""),
        )
    )
    if "anthropic" in hay or "claude" in hay:
        return Provider.ANTHROPIC
    if "openai" in hay or "gpt" in hay or hay.startswith("o"):
        return Provider.OPENAI
    if "ollama" in hay:
        return Provider.OLLAMA
    if "bedrock" in hay:
        return Provider.BEDROCK
    if "vertex" in hay or "gemini" in hay:
        return Provider.VERTEX
    return Provider.OTHER


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class StepCostCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records a StepCost span per run."""

    # Callbacks may fire on worker threads (batch/parallel execution).
    run_inline = True

    def __init__(self, client: StepCost) -> None:
        self._client = client
        self._runs: dict[UUID, ActiveSpan] = {}
        # run_id -> parent span_id for passthrough runs that create no span
        self._passthrough: dict[UUID, str | None] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Run-tree helpers
    # ------------------------------------------------------------------ #
    def _parent_span_id(self, parent_run_id: UUID | None) -> str | None:
        with self._lock:
            if parent_run_id is None:
                return None
            active = self._runs.get(parent_run_id)
            if active is not None:
                return active.span.span_id
            return self._passthrough.get(parent_run_id)

    def _open(
        self,
        run_id: UUID,
        parent_run_id: UUID | None,
        *,
        kind: SpanKind,
        name: str | None,
        model: str | None = None,
        provider: Provider | None = None,
    ) -> None:
        active = self._client.start_span(
            kind=kind,
            name=name,
            model=model,
            provider=provider,
            parent_span_id=self._parent_span_id(parent_run_id),
        )
        with self._lock:
            self._runs[run_id] = active

    def _close(self, run_id: UUID, *, error: BaseException | None = None) -> None:
        with self._lock:
            active = self._runs.pop(run_id, None)
            self._passthrough.pop(run_id, None)
        if active is None:
            return
        if error is not None:
            active.set_metadata(error=type(error).__name__)
        active.finish()

    @staticmethod
    def _run_name(serialized: dict | None, kwargs: dict) -> str | None:
        name = kwargs.get("name")
        if not name and serialized:
            name = serialized.get("name") or (serialized.get("id") or [None])[-1]
        return name

    # ------------------------------------------------------------------ #
    # Chains (agent steps)
    # ------------------------------------------------------------------ #
    def on_chain_start(
        self,
        serialized: dict | None,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = self._run_name(serialized, kwargs)
        if not name or name in _PASSTHROUGH_NAMES or name.startswith(_PASSTHROUGH_PREFIXES):
            # Structural runnable — no span; children parent through it.
            parent = self._parent_span_id(parent_run_id)
            with self._lock:
                self._passthrough[run_id] = parent
            return
        self._open(run_id, parent_run_id, kind=SpanKind.AGENT_STEP, name=name)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)

    # ------------------------------------------------------------------ #
    # LLM calls
    # ------------------------------------------------------------------ #
    def _on_model_start(
        self,
        serialized: dict | None,
        *,
        run_id: UUID,
        parent_run_id: UUID | None,
        **kwargs: Any,
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or self._run_name(
            serialized, kwargs
        )
        self._open(
            run_id,
            parent_run_id,
            kind=SpanKind.LLM_GENERATION,
            name=model,
            model=model,
            provider=_guess_provider(serialized, params),
        )

    def on_llm_start(
        self,
        serialized: dict | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._on_model_start(serialized, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_chat_model_start(
        self,
        serialized: dict | None,
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._on_model_start(serialized, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            active = self._runs.get(run_id)
        if active is not None:
            usage = self._extract_usage(response)
            if usage:
                active.record_usage(**usage)
        self._close(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int] | None:
        """Invoice-safe usage from an LLMResult.

        Prefers the provider-agnostic ``usage_metadata`` on the AIMessage
        (LangChain aggregates cache reads/writes INTO input_tokens there, so we
        subtract them back out); falls back to ``llm_output`` token dicts.
        """
        # 1) usage_metadata on the first generation's message
        try:
            gen = response.generations[0][0]
            meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
        except (AttributeError, IndexError, TypeError):
            meta = None
        if meta:
            in_total = _int(meta.get("input_tokens"))
            out_total = _int(meta.get("output_tokens"))
            in_det = meta.get("input_token_details") or {}
            out_det = meta.get("output_token_details") or {}
            cache_read = _int(in_det.get("cache_read"))
            cache_write = _int(in_det.get("cache_creation"))
            reasoning = _int(out_det.get("reasoning"))
            uncached = max(in_total - cache_read - cache_write, 0)
            return {
                "input_tokens": uncached,
                "output_tokens": max(out_total - reasoning, 0),
                "cached_input_tokens": cache_read,
                "cache_creation_tokens": cache_write,
                "reasoning_tokens": reasoning,
            }

        # 2) llm_output fallbacks (OpenAI "token_usage", Anthropic "usage")
        llm_output = getattr(response, "llm_output", None) or {}
        tu = llm_output.get("token_usage") or {}
        if tu:
            prompt = _int(tu.get("prompt_tokens"))
            completion = _int(tu.get("completion_tokens"))
            cached = _int((tu.get("prompt_tokens_details") or {}).get("cached_tokens"))
            reasoning = _int(
                (tu.get("completion_tokens_details") or {}).get("reasoning_tokens")
            )
            return {
                "input_tokens": max(prompt - cached, 0),
                "output_tokens": max(completion - reasoning, 0),
                "cached_input_tokens": cached,
                "reasoning_tokens": reasoning,
            }
        au = llm_output.get("usage") or {}
        if au:
            return {
                "input_tokens": _int(au.get("input_tokens")),
                "output_tokens": _int(au.get("output_tokens")),
                "cached_input_tokens": _int(au.get("cache_read_input_tokens")),
                "cache_creation_tokens": _int(au.get("cache_creation_input_tokens")),
            }
        return None

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def on_tool_start(
        self,
        serialized: dict | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open(
            run_id,
            parent_run_id,
            kind=SpanKind.TOOL_CALL,
            name=self._run_name(serialized, kwargs),
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)

    # ------------------------------------------------------------------ #
    # Retrievers
    # ------------------------------------------------------------------ #
    def on_retriever_start(
        self,
        serialized: dict | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._open(
            run_id,
            parent_run_id,
            kind=SpanKind.RETRIEVAL,
            name=self._run_name(serialized, kwargs),
        )

    def on_retriever_end(self, documents: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id)

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close(run_id, error=error)
