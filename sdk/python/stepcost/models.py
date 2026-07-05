"""Pydantic models for StepCost spans and cost ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SpanKind(StrEnum):
    TRACE = "trace"
    AGENT_STEP = "agent_step"
    LLM_GENERATION = "llm_generation"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    RETRY = "retry"


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    OLLAMA = "ollama"
    OTHER = "other"


class PayloadCapture(StrEnum):
    NONE = "none"
    HASH = "hash"
    FULL = "full"


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    cache_creation_tokens: int = 0
    # Anthropic 1h-TTL cache writes bill at 2x input (5m-TTL writes, above,
    # bill at 1.25x) — collapsing the two undercounts 1h traffic by 37.5%.
    cache_creation_1h_tokens: int = 0
    embedding_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cached_input_tokens
            + self.reasoning_tokens
            + self.cache_creation_tokens
            + self.cache_creation_1h_tokens
            + self.embedding_tokens
        )


class CostBreakdown(BaseModel):
    input_usd: Decimal = Decimal("0")
    output_usd: Decimal = Decimal("0")
    cached_input_usd: Decimal = Decimal("0")
    reasoning_usd: Decimal = Decimal("0")
    cache_creation_usd: Decimal = Decimal("0")
    embedding_usd: Decimal = Decimal("0")
    total_usd: Decimal = Decimal("0")
    price_table_version: str | None = None

    @field_validator(
        "input_usd",
        "output_usd",
        "cached_input_usd",
        "reasoning_usd",
        "cache_creation_usd",
        "embedding_usd",
        "total_usd",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        return Decimal(str(value))


class SpanPayload(BaseModel):
    capture: PayloadCapture = PayloadCapture.NONE
    prompt_hash: str | None = None
    response_hash: str | None = None


class Span(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid4()))
    trace_id: str
    parent_span_id: str | None = None

    kind: SpanKind
    name: str | None = None

    org_id: str | None = None
    project_id: str
    customer_id: str | None = None
    feature_id: str | None = None
    team_id: str | None = None
    environment: str = "dev"

    provider: Provider | None = None
    model: str | None = None
    prompt_version: str | None = None

    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost: CostBreakdown = Field(default_factory=CostBreakdown)

    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: int = 0

    session_id: str | None = None
    user_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    payload: SpanPayload = Field(default_factory=SpanPayload)

    def with_cost(self, cost: CostBreakdown) -> Span:
        return self.model_copy(update={"cost": cost})


class ModelPricing(BaseModel):
    # Frozen: the default table is a process-wide lru_cache'd singleton, so a
    # mutable rate here would let one caller silently repoison all pricing.
    model_config = ConfigDict(frozen=True)

    provider: Provider
    input_per_1m: Decimal
    output_per_1m: Decimal
    cached_input_per_1m: Decimal | None = None
    reasoning_per_1m: Decimal | None = None
    cache_write_per_1m: Decimal | None = None
    cache_write_1h_per_1m: Decimal | None = None
    embedding_per_1m: Decimal | None = None


class PriceTable(BaseModel):
    version: str
    models: dict[str, ModelPricing]
