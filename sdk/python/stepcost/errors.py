"""StepCost errors."""


class StepCostError(Exception):
    """Base error."""


class ContextRequiredError(StepCostError):
    """Raised when agent_step/llm_call used outside an active trace."""


class TraceCostUnavailableError(StepCostError):
    """Raised when trace cost cannot be computed for the active client."""
