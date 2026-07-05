"""Invoice reconciliation gate helpers."""

from __future__ import annotations

from decimal import Decimal


def reconciliation_error_pct(computed: Decimal, billed: Decimal) -> Decimal:
    if billed <= 0:
        raise ValueError("billed amount must be positive")
    return abs(computed - billed) / billed


def passes_reconciliation_gate(
    computed: Decimal,
    billed: Decimal,
    *,
    tolerance: Decimal = Decimal("0.02"),
) -> bool:
    return reconciliation_error_pct(computed, billed) <= tolerance
