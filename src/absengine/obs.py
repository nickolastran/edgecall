"""Stage observability: every stage logs rows in/out and wall time."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

log = logging.getLogger("absengine")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "pytensor", "pymc", "arviz"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def stage(name: str, rows_in: int | None = None) -> Iterator[dict[str, int]]:
    """`with stage("re24", len(df)) as s: ...; s["out"] = len(result)`."""
    t0 = time.perf_counter()
    info: dict[str, int] = {}
    log.info("[%s] start rows_in=%s", name, rows_in)
    yield info
    log.info(
        "[%s] done rows_in=%s rows_out=%s %.1fs",
        name,
        rows_in,
        info.get("out"),
        time.perf_counter() - t0,
    )
