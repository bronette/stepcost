"""Span sinks — local persistence and cloud ingest."""

from stepcost.sinks.base import SpanSink
from stepcost.sinks.sqlite import SQLiteSink
from stepcost.sinks.stdout import StdoutSink

__all__ = ["SpanSink", "SQLiteSink", "StdoutSink"]
