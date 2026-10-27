"""Shared exceptions and user-facing error text."""

from __future__ import annotations

import re

_PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:/|\\)(?:[\w .\-]+[/\\])+[\w .\-]*")


class InsulinMatrixError(Exception):
    """Base error for expected application failures."""


class ExcelError(InsulinMatrixError):
    """The workbook could not be read or written."""


class UnitError(InsulinMatrixError):
    """A measurement unit is missing or unsupported."""


class ModelingError(InsulinMatrixError):
    """Training or evaluation cannot be performed as requested."""


class PredictionError(InsulinMatrixError):
    """A saved model cannot score the supplied rows."""


class StorageError(InsulinMatrixError):
    """A local file or database operation was refused or failed."""


def public_message(exc: BaseException) -> str:
    """Return a short message without filesystem locations."""
    text = str(exc).strip() or exc.__class__.__name__
    text = _PATH_RE.sub("[path]", text)
    if len(text) > 600:
        text = text[:600] + "…"
    return text
