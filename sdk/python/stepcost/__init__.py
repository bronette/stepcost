"""StepCost — FinOps for LLM agents."""

from stepcost.client import StepCost
from stepcost.context import ActiveSpan, TraceContext, agent_step, llm_call
from stepcost.errors import (
    ContextRequiredError,
    StepCostError,
    TraceCostUnavailableError,
)
from stepcost.extractors import extract_usage
from stepcost.extractors.openai import extract_openai_usage, naive_openai_usage
from stepcost.models import (
    CostBreakdown,
    Provider,
    Span,
    SpanKind,
    TokenUsage,
)
from stepcost.pricing import compute_cost, cost_for_model, default_price_table

__all__ = [
    "ActiveSpan",
    "ContextRequiredError",
    "CostBreakdown",
    "StepCost",
    "StepCostError",
    "Provider",
    "Span",
    "SpanKind",
    "TokenUsage",
    "TraceContext",
    "TraceCostUnavailableError",
    "agent_step",
    "compute_cost",
    "cost_for_model",
    "default_price_table",
    "extract_openai_usage",
    "extract_usage",
    "llm_call",
    "naive_openai_usage",
]

__version__ = "0.3.2"
