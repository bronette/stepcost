"""Anthropic client wrapper — zero lines per call site, no framework needed.

Usage::

    import anthropic
    from stepcost import StepCost
    from stepcost.integrations.anthropic import instrument_anthropic

    cc = StepCost(project="my-app", sink="sqlite:///~/.stepcost/my-app.db")
    client = instrument_anthropic(anthropic.Anthropic(), cc)

    with cc.trace(feature_id="chat") as trace:
        client.messages.create(model="claude-haiku-4-5", max_tokens=64, messages=[...])
    print(trace.total_usd)

Wraps ``messages.create`` in place. Each call becomes a priced span with
invoice-safe extraction (cache read vs 5m/1h cache-write TTLs split); an
exception still closes the span with an error tag.
"""

from __future__ import annotations

import functools
from typing import Any

from stepcost.client import StepCost
from stepcost.models import Provider, SpanKind


def instrument_anthropic(client: Any, cc: StepCost) -> Any:
    """Wrap an ``anthropic.Anthropic`` client's ``messages.create`` in place."""
    messages = getattr(client, "messages", None)
    if messages is None or not hasattr(messages, "create"):
        return client
    create = messages.create

    @functools.wraps(create)
    def traced(*args: Any, **kwargs: Any) -> Any:
        model = str(kwargs.get("model") or "")
        with cc.span(
            kind=SpanKind.LLM_GENERATION,
            name=model or None,
            model=model or None,
            provider=Provider.ANTHROPIC,
        ) as sp:
            try:
                response = create(*args, **kwargs)
            except BaseException as exc:
                sp.set_metadata(error=type(exc).__name__)
                raise
            try:
                sp.record(response, provider=Provider.ANTHROPIC)
            except ValueError:
                sp.set_metadata(usage="unavailable")
            return response

    messages.create = traced
    return client
