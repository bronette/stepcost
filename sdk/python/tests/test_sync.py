"""Tests for provider cost sync + reconciliation (fixtures, no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from stepcost import StepCost, Provider, llm_call
from stepcost.cli import main as cli_main
from stepcost.report import build_reconciliation, build_summary
from stepcost.sync import (
    ProviderCost,
    load_provider_costs,
    parse_anthropic_cost_page,
    parse_openai_cost_page,
    store_provider_costs,
)

ANTHROPIC_PAGE = {
    "data": [
        {
            "starting_at": "2026-07-08T00:00:00Z",
            "ending_at": "2026-07-09T00:00:00Z",
            "results": [
                {
                    "amount": "1.6070",  # CENTS as decimal string -> $0.016070
                    "currency": "USD",
                    "cost_type": "tokens",
                    "description": "Claude Haiku 4.5 Usage - Input Tokens",
                    "model": "claude-haiku-4-5",
                    "token_type": "uncached_input_tokens",
                    "service_tier": "standard",
                    "workspace_id": None,
                },
                {
                    "amount": "0.2943",  # $0.002943
                    "currency": "USD",
                    "cost_type": "tokens",
                    "description": "Claude Haiku 4.5 Usage - Cache Write",
                    "model": "claude-haiku-4-5",
                    "token_type": "cache_creation.ephemeral_5m_input_tokens",
                },
            ],
        }
    ],
    "has_more": False,
    "next_page": None,
}

OPENAI_PAGE = {
    "object": "page",
    "data": [
        {
            "object": "bucket",
            "start_time": int(datetime(2026, 7, 8, tzinfo=timezone.utc).timestamp()),
            "end_time": int(datetime(2026, 7, 9, tzinfo=timezone.utc).timestamp()),
            "results": [
                {
                    "object": "organization.costs.result",
                    "amount": {"value": 0.0132, "currency": "usd"},  # DOLLARS
                    "line_item": "gpt-4o-mini, input",
                    "project_id": None,
                }
            ],
        }
    ],
    "has_more": False,
    "next_page": None,
}


def test_anthropic_cents_converted_to_dollars():
    records = parse_anthropic_cost_page(ANTHROPIC_PAGE)
    assert len(records) == 2
    assert records[0].day == "2026-07-08"
    assert records[0].amount_usd == Decimal("0.016070")
    assert records[0].model == "claude-haiku-4-5"
    assert records[1].token_type == "cache_creation.ephemeral_5m_input_tokens"


def test_openai_dollars_kept_as_dollars():
    records = parse_openai_cost_page(OPENAI_PAGE)
    assert len(records) == 1
    assert records[0].day == "2026-07-08"
    assert records[0].amount_usd == Decimal("0.0132")
    assert records[0].model == "gpt-4o-mini"


def test_store_is_idempotent_per_provider_day(tmp_path: Path):
    db = tmp_path / "s.db"
    records = parse_anthropic_cost_page(ANTHROPIC_PAGE)
    store_provider_costs(db, records)
    store_provider_costs(db, records)  # re-sync must not duplicate
    loaded = load_provider_costs(db)
    assert len(loaded) == 2
    assert sum(r.amount_usd for r in loaded) == Decimal("0.019013")


def test_load_without_table_returns_empty(tmp_path: Path):
    db = tmp_path / "empty.db"
    import sqlite3

    sqlite3.connect(str(db)).close()
    assert load_provider_costs(db) == []


def _spans_costing(cc: StepCost, usd_target_tokens: int) -> None:
    with cc.trace():
        with llm_call(model="claude-haiku-4-5", provider=Provider.ANTHROPIC) as call:
            call.record_usage(input_tokens=usd_target_tokens)


def test_reconciliation_drift_and_coverage(tmp_path: Path):
    db = tmp_path / "r.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    # SDK observes $0.019 today: 19_000 input tokens @ $1/1M on haiku-4-5
    _spans_costing(cc, 19_000)
    cc.flush()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    provider = [
        ProviderCost(provider="anthropic", day=today, amount_usd=Decimal("0.019013")),
        # A second day where the SDK saw nothing — pure uninstrumented spend
        ProviderCost(provider="anthropic", day="2026-01-01", amount_usd=Decimal("1.00")),
    ]
    from stepcost.report import load_spans

    spans = load_spans(str(db))
    rows = build_reconciliation(spans, provider)
    by_day = {r.day: r for r in rows}

    r_today = by_day[today]
    assert r_today.sdk_usd == Decimal("0.019")
    assert r_today.provider_usd == Decimal("0.019013")
    assert r_today.drift_pct < 0.1  # within the 2% gate
    assert 99.0 < r_today.coverage_pct <= 100.0

    r_gap = by_day["2026-01-01"]
    assert r_gap.sdk_usd == 0
    assert r_gap.coverage_pct == 0.0  # bill exists, SDK saw none of it

    summary = build_summary(spans, provider_costs=provider)
    assert len(summary.reconciliation) == 2


def test_cli_report_shows_reconciliation_section(tmp_path: Path, capsys):
    db = tmp_path / "c.db"
    cc = StepCost(project="demo", sink=f"sqlite:///{db}")
    _spans_costing(cc, 19_000)
    cc.flush()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    store_provider_costs(
        db, [ProviderCost(provider="anthropic", day=today, amount_usd=Decimal("0.019013"))]
    )

    assert cli_main(["report", str(db)]) == 0
    out = capsys.readouterr().out
    assert "Provider reconciliation" in out
    assert "2% gate" in out
    assert "PASS" in out

    assert cli_main(["report", str(db), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["reconciliation"][0]["provider"] == "anthropic"
    assert payload["reconciliation"][0]["drift_pct"] < 0.1


def test_sync_cli_requires_admin_key(tmp_path: Path, capsys, monkeypatch):
    db = tmp_path / "k.db"
    db.touch()
    monkeypatch.delenv("ANTHROPIC_ADMIN_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-not-an-admin-key")
    assert cli_main(["sync", "anthropic", str(db)]) == 2
    assert "sk-ant-admin" in capsys.readouterr().err
