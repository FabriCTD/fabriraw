"""FabriRAW (.fraw / .frw / .fset) — reference Python implementation.

Pure Python + numpy; no rawpy / PySide6 imports here (those are optional extras).

Public names are resolved lazily (PEP 562) so that ``python -m fraw.format``
runs without the "found in sys.modules" runpy warning.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

_EXPORTS = {
    "MAGIC": "format",
    "SET_MAGIC": "format",
    "Color": "format",
    "ErrorCode": "format",
    "ExtraChunk": "format",
    "FrawError": "format",
    "FrawFile": "format",
    "Head": "format",
    "Thumbnail": "format",
    "update_dev": "format",
    "FrawSet": "fset",
}

__all__ = sorted([*_EXPORTS, "__version__"])

if TYPE_CHECKING:  # pragma: no cover
    from .format import (  # noqa: F401
        MAGIC,
        SET_MAGIC,
        Color,
        ErrorCode,
        ExtraChunk,
        FrawError,
        FrawFile,
        Head,
        Thumbnail,
        update_dev,
    )
    from .fset import FrawSet  # noqa: F401


def __getattr__(name: str) -> Any:
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'fraw' has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{mod}", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return __all__
