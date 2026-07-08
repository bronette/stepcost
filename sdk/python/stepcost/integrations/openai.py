"""OpenAI client wrapper — zero lines per call site, no framework needed.

Usage::

    from openai import OpenAI
    from stepcost import StepCost
    from stepcost.integrations.openai import instrument_openai

    cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")
    client = instrument_openai(OpenAI(), cc)

    with cc.trace(feature_id="chat") as trace:
        client.chat.completions.create(model="gpt-4o-mini", messages=[...])
    print(trace.total_usd)

Wraps ``chat.completions.create``, ``responses.create``, and
``embeddings.create`` in place. Each call becomes a priced span (invoice-safe
extraction via ``ActiveSpan.record``); exceptions still close the span with an
error tag. Works with any object shaped like the official SDK client.
"""

from __future__ import annotations

import functools
from typing import Any

from stepcost.client import StepCost
from stepcost.models import Provider, SpanKind


def _wrap_create(create: Any, cc: StepCost, *, kind: SpanKind, provider: Provider) -> Any:
    @functools.wraps(create)
    def traced(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "")
        with cc.span(kind=kind, name=model or None, model=model or None, provider=provider) as sp:
            try:
                response = create(*args, **kwargs)
            except BaseException as exc:
                sp.set_metadata(error=type(exc).__name__)
                raise
            try:
                sp.record(response, provider=provider)
            except ValueError:
                # Streaming responses / shapes without usage: leave tokens at 0
                # rather than failing the caller's request.
                sp.set_metadata(usage="unavailable")
            return response

    return traced


def instrument_openai(client: Any, cc: StepCost) -> Any:
    """Wrap an ``openai.OpenAI`` client's create methods in place."""
    chat = getattr(getattr(client, "chat", None), "completions", None)
    if chat is not None and hasattr(chat, "create"):
        chat.create = _wrap_create(
            chat.create, cc, kind=SpanKind.LLM_GENERATION, provider=Provider.OPENAI
        )
    responses = getattr(client, "responses", None)
    if responses is not None and hasattr(responses, "create"):
        responses.create = _wrap_create(
            responses.create, cc, kind=SpanKind.LLM_GENERATION, provider=Provider.OPENAI
        )
    embeddings = getattr(client, "embeddings", None)
    if embeddings is not None and hasattr(embeddings, "create"):
        embeddings.create = _wrap_create(
            embeddings.create, cc, kind=SpanKind.EMBEDDING, provider=Provider.OPENAI
        )
    return client
