"""Stdout sink for local debugging."""

from __future__ import annotations

import json
import sys

from stepcost.models import Span


class StdoutSink:
    def emit(self, spans: list[Span]) -> None:
        for span in spans:
            print(json.dumps(span.model_dump(mode="json"), default=str))

    def flush(self) -> None:
        # stdout may be block-buffered when piped; without this, spans emitted
        # shortly before an abrupt exit never reach the pipe.
        sys.stdout.flush()
