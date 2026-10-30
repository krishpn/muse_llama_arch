#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Shared output-persistence helper for every evaluation/*.py script -- avoids duplicating the
same tee-stdout-and-json-dump boilerplate six times. Every script keeps its existing
"compute + print + flag-if-notable" shape; this only adds a `captured_output()` wrapper around
main()'s body and a single `save_report()` call at the end, so what's on disk is exactly what was
printed, plus a structured JSON of whatever metrics the section functions already return.

Not run directly -- imported by the other evaluation/*.py scripts.
"""

import contextlib
import io
import json
import sys
from collections.abc import Iterator
from pathlib import Path


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def captured_output() -> Iterator[io.StringIO]:
    """Tees everything printed inside the `with` block to both the real stdout (interactive runs
    behave exactly as before) and a buffer, so callers can persist what was printed without
    restructuring every print call into a return value."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        yield buf


def save_report(out_dir: Path, name: str, text: str, metrics: dict | None) -> None:
    """Writes {name}_report.txt always, and {name}_metrics.json when metrics is not None -- the
    one place that creates the directory and names the two output files, so every script's
    main() ends with an identical call instead of six slightly-different ones.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{name}_report.txt"
    report_path.write_text(text)
    print(f"saved: {report_path}")
    if metrics is not None:
        metrics_path = out_dir / f"{name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"saved: {metrics_path}")
