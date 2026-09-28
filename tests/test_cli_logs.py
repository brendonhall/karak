"""karak log records reach the reporter only while a run is captured."""

from __future__ import annotations

import logging

import pytest

from karak.cli.logs import capture_logs


class Recorder:
    def __init__(self):
        self.lines = []

    def log(self, level, msg):
        self.lines.append((level, msg))


def test_capture_forwards_info_and_above():
    rec = Recorder()
    logger = logging.getLogger("karak.io.loaders")
    with capture_logs(rec):
        logger.info("hello %s", "x")
        logger.warning("careful")
        logger.debug("hidden")
    logger.info("after the run")
    assert rec.lines == [("info", "hello x"), ("warning", "careful")]


def test_capture_restores_logger_state_after_error():
    karak_logger = logging.getLogger("karak")
    before = (list(karak_logger.handlers), karak_logger.level,
              karak_logger.propagate)
    with pytest.raises(RuntimeError):
        with capture_logs(Recorder()):
            raise RuntimeError("stage blew up")
    after = (list(karak_logger.handlers), karak_logger.level,
             karak_logger.propagate)
    assert after == before
