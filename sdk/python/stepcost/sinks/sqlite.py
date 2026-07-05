"""SQLite sink for offline / dogfood persistence."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from urllib.parse import urlparse

from stepcost.models import Span


def sqlite_url_to_path(url: str) -> Path:
    """Resolve sqlite:///path, sqlite://relative/path, and ~ forms to a Path."""
    parsed = urlparse(url)
    if parsed.scheme != "sqlite":
        raise ValueError(f"Expected sqlite:// URL, got {url!r}")
    raw = (parsed.netloc or "") + (parsed.path or "")
    if raw.startswith("/~"):
        raw = raw[1:]
    return Path(raw).expanduser()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    parent_span_id TEXT,
    kind TEXT NOT NULL,
    project_id TEXT NOT NULL,
    feature_id TEXT,
    customer_id TEXT,
    model TEXT,
    total_usd TEXT,
    started_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_project ON spans(project_id);
"""


class SQLiteSink:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # Flushes can arrive from any thread; the lock serializes all access
        # since a shared sqlite3 connection is not itself thread-safe.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)

    @classmethod
    def from_url(cls, url: str) -> SQLiteSink:
        return cls(sqlite_url_to_path(url))

    def emit(self, spans: list[Span]) -> None:
        rows = [
            (
                span.span_id,
                span.trace_id,
                span.parent_span_id,
                span.kind.value,
                span.project_id,
                span.feature_id,
                span.customer_id,
                span.model,
                str(span.cost.total_usd),
                span.started_at.isoformat(),
                json.dumps(span.model_dump(mode="json"), default=str),
            )
            for span in spans
        ]
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO spans (
                    span_id, trace_id, parent_span_id, kind, project_id,
                    feature_id, customer_id, model, total_usd, started_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

    def flush(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def trace_total_usd(self, trace_id: str) -> float:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(CAST(total_usd AS REAL)), 0) FROM spans WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
        return float(row[0] if row else 0.0)
