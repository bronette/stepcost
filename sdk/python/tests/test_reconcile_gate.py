"""Tests for invoice reconciliation gate."""

from decimal import Decimal

from stepcost.reconcile_gate import passes_reconciliation_gate, reconciliation_error_pct


def test_gate_pass_within_two_percent():
    assert passes_reconciliation_gate(Decimal("1.00"), Decimal("1.01")) is True
    assert passes_reconciliation_gate(Decimal("0.98"), Decimal("1.00")) is True
    assert reconciliation_error_pct(Decimal("0.98"), Decimal("1.00")) == Decimal("0.02")


def test_gate_fail_over_two_percent():
    assert passes_reconciliation_gate(Decimal("1.10"), Decimal("1.00")) is False
