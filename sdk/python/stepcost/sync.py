"""Provider-side cost ingestion — the other half of the ledger.

Pulls what the provider says it will bill (org Usage & Cost Admin APIs) into
the same SQLite file the SDK writes spans to. ``stepcost report`` then shows a
reconciliation section: SDK-observed $ vs provider-reported $, per day —
continuous invoice reconciliation plus a coverage number that exposes
uninstrumented spend.

    stepcost sync anthropic ~/.stepcost/app.db --days 7   # needs ANTHROPIC_ADMIN_KEY
    stepcost sync openai    ~/.stepcost/app.db --days 7   # needs OPENAI_ADMIN_KEY

Provider APIs report org/day/model granularity — they can never see agent
steps, features, or customers. The SDK can't see spend it didn't wrap. Each
side audits the other's blind spot.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

_ANTHROPIC_COST_URL = "https://api.anthropic.com/v1/organizations/cost_report"
_OPENAI_COST_URL = "https://api.openai.com/v1/organization/costs"
_TIMEOUT = 30


class ProviderCost(BaseModel):
    provider: str  # "anthropic" | "openai"
    day: str  # YYYY-MM-DD (UTC bucket start)
    model: str = ""  # empty when the provider doesn't attribute to a model
    token_type: str = ""
    cost_type: str = ""
    description: str = ""
    amount_usd: Decimal


def _get_json(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


# --------------------------------------------------------------------------- #
# Anthropic — /v1/organizations/cost_report (admin key sk-ant-admin01-...)
# --------------------------------------------------------------------------- #
def parse_anthropic_cost_page(page: dict) -> list[ProviderCost]:
    out: list[ProviderCost] = []
    for bucket in page.get("data", []):
        day = str(bucket.get("starting_at", ""))[:10]
        for r in bucket.get("results", []):
            # amount is a decimal string in CENTS ("123.45" == $1.2345)
            cents = Decimal(str(r.get("amount", "0")))
            out.append(
                ProviderCost(
                    provider="anthropic",
                    day=day,
                    model=r.get("model") or "",
                    token_type=r.get("token_type") or "",
                    cost_type=r.get("cost_type") or "",
                    description=r.get("description") or "",
                    amount_usd=cents / Decimal("100"),
                )
            )
    return out


def fetch_anthropic_costs(
    admin_key: str, *, starting_at: str, ending_at: str
) -> list[ProviderCost]:
    headers = {
        "x-api-key": admin_key,
        "anthropic-version": "2023-06-01",
        "User-Agent": "stepcost (https://stepcost.com)",
    }
    params = {
        "starting_at": starting_at,
        "ending_at": ending_at,
        "bucket_width": "1d",
        "group_by[]": "description",
    }
    records: list[ProviderCost] = []
    page_token: str | None = None
    while True:
        q = dict(params)
        if page_token:
            q["page"] = page_token
        page = _get_json(f"{_ANTHROPIC_COST_URL}?{urllib.parse.urlencode(q)}", headers)
        records.extend(parse_anthropic_cost_page(page))
        if not page.get("has_more"):
            return records
        page_token = page.get("next_page")


# --------------------------------------------------------------------------- #
# OpenAI — /v1/organization/costs (admin key)
# --------------------------------------------------------------------------- #
def parse_openai_cost_page(page: dict) -> list[ProviderCost]:
    out: list[ProviderCost] = []
    for bucket in page.get("data", []):
        start = bucket.get("start_time", 0)
        day = datetime.fromtimestamp(int(start), tz=timezone.utc).strftime("%Y-%m-%d")
        for r in bucket.get("results", []):
            amount = r.get("amount") or {}
            # amount.value is in DOLLARS (unlike Anthropic's cents)
            value = Decimal(str(amount.get("value", "0")))
            line_item = r.get("line_item") or ""
            out.append(
                ProviderCost(
                    provider="openai",
                    day=day,
                    # line_item looks like "gpt-4o-mini, input" — best-effort model
                    model=line_item.split(",")[0].strip() if line_item else "",
                    description=line_item,
                    amount_usd=value,
                )
            )
    return out


def fetch_openai_costs(admin_key: str, *, start_time: int, end_time: int) -> list[ProviderCost]:
    headers = {
        "Authorization": f"Bearer {admin_key}",
        "User-Agent": "stepcost (https://stepcost.com)",
    }
    params: dict[str, str] = {
        "start_time": str(start_time),
        "end_time": str(end_time),
        "bucket_width": "1d",
        "group_by": "line_item",
        "limit": "31",
    }
    records: list[ProviderCost] = []
    page_token: str | None = None
    while True:
        q = dict(params)
        if page_token:
            q["page"] = page_token
        page = _get_json(f"{_OPENAI_COST_URL}?{urllib.parse.urlencode(q)}", headers)
        records.extend(parse_openai_cost_page(page))
        if not page.get("has_more"):
            return records
        page_token = page.get("next_page")


# --------------------------------------------------------------------------- #
# Storage — provider_costs table in the same sink DB
# --------------------------------------------------------------------------- #
_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_costs (
    provider TEXT NOT NULL,
    day TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    token_type TEXT NOT NULL DEFAULT '',
    cost_type TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    amount_usd TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_provider_costs_day ON provider_costs(provider, day);
"""


def store_provider_costs(db_path: Path, records: list[ProviderCost]) -> int:
    """Idempotent: replaces all rows for each (provider, day) present in records."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        fetched_at = datetime.now(timezone.utc).isoformat()
        days = {(r.provider, r.day) for r in records}
        for provider, day in days:
            conn.execute(
                "DELETE FROM provider_costs WHERE provider = ? AND day = ?", (provider, day)
            )
        conn.executemany(
            "INSERT INTO provider_costs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (r.provider, r.day, r.model, r.token_type, r.cost_type,
                 r.description, str(r.amount_usd), fetched_at)
                for r in records
            ],
        )
        conn.commit()
        return len(records)
    finally:
        conn.close()


def load_provider_costs(db_path: Path) -> list[ProviderCost]:
    conn = sqlite3.connect(str(db_path))
    try:
        try:
            rows = conn.execute(
                "SELECT provider, day, model, token_type, cost_type, description, amount_usd "
                "FROM provider_costs"
            ).fetchall()
        except sqlite3.OperationalError:
            return []  # table absent — sync never ran
    finally:
        conn.close()
    return [
        ProviderCost(
            provider=p, day=d, model=m, token_type=tt, cost_type=ct,
            description=desc, amount_usd=Decimal(a),
        )
        for p, d, m, tt, ct, desc, a in rows
    ]


def default_window(days: int) -> tuple[str, str, int, int]:
    """(anthropic starting_at, ending_at RFC3339; openai start, end unix)."""
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    start = end - timedelta(days=days + 1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return start.strftime(fmt), end.strftime(fmt), int(start.timestamp()), int(end.timestamp())
