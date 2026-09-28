"""Route karak log records to a reporter for the duration of a run."""

from __future__ import annotations

import logging
from contextlib import contextmanager


class ReporterLogHandler(logging.Handler):
    def __init__(self, reporter):
        super().__init__(level=logging.INFO)
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.reporter.log(record.levelname.lower(), record.getMessage())
        except Exception:
            self.handleError(record)


@contextmanager
def capture_logs(reporter, logger_name: str = "karak"):
    """Send ``karak`` INFO+ records to ``reporter.log`` while the block runs.

    Propagation is switched off meanwhile so records are not printed a
    second time underneath a live display; everything is restored on exit.
    """
    logger = logging.getLogger(logger_name)
    handler = ReporterLogHandler(reporter)
    saved = (logger.level, logger.propagate)
    logger.addHandler(handler)
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved[0])
        logger.propagate = saved[1]
