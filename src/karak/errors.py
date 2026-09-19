"""Shared error types for the numeric core, stages, and flow layers."""

from __future__ import annotations


class StageError(Exception):
    """Raised when a stage receives invalid inputs or parameters."""
