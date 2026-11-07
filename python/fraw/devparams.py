"""Non-destructive development parameters (SPEC §4.6, chunk ``DEV ``).

:class:`DevParams` mirrors the DEV JSON object key by key. Parsing is lenient
and never raises on content: missing keys take the default, numbers outside the
normative range are clamped, values of the wrong type (including ``bool`` for
numeric fields and non-finite floats) fall back to the default, and unknown
keys are ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = ["DevParams", "DEV_VERSION", "RANGES", "WB_MODES", "GAMMA_MODES", "DEMOSAIC_METHODS"]

DEV_VERSION = 1

WB_MODES = ("as_shot", "daylight", "custom")
GAMMA_MODES = ("srgb", "power", "linear")
DEMOSAIC_METHODS = ("bilinear", "ahd")

# Normative ranges (SPEC §4.6). ``gamma`` and ``wb_custom`` have no range in the
# SPEC; these are the implementation limits (documented, conservative).
RANGES: dict[str, tuple[float, float]] = {
    "exposure_ev": (-4.0, 4.0),
    "temperature_shift": (-1.0, 1.0),
    "tint_shift": (-1.0, 1.0),
    "contrast": (-1.0, 1.0),
    "saturation": (0.0, 2.0),
    "highlights": (-1.0, 1.0),
    "shadows": (-1.0, 1.0),
    "gamma": (1.0, 4.0),
}
WB_CUSTOM_RANGE = (1e-3, 64.0)  # per-channel multiplier limits

_DEFAULT_WB_CUSTOM = (2.0, 1.0, 1.5)


def _num(value: Any) -> float | None:
    """Finite real number (not bool) → float, else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        v = float(value)
    except (OverflowError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass
class DevParams:
    exposure_ev: float = 0.0
    wb_mode: str = "as_shot"
    wb_custom: tuple[float, float, float] = field(default=_DEFAULT_WB_CUSTOM)
    temperature_shift: float = 0.0
    tint_shift: float = 0.0
    contrast: float = 0.0
    saturation: float = 1.0
    highlights: float = 0.0
    shadows: float = 0.0
    gamma_mode: str = "srgb"
    gamma: float = 2.2
    demosaic: str = "ahd"

    # ------------------------------------------------------------------ parse
    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "DevParams":
        p = cls()
        if not isinstance(d, Mapping):
            return p
        for key, (lo, hi) in RANGES.items():
            if key in d:
                v = _num(d[key])
                if v is not None:
                    setattr(p, key, _clamp(v, lo, hi))
        for key, allowed in (("wb_mode", WB_MODES), ("gamma_mode", GAMMA_MODES), ("demosaic", DEMOSAIC_METHODS)):
            v = d.get(key)
            if isinstance(v, str) and v in allowed:
                setattr(p, key, v)
        wc = d.get("wb_custom")
        if isinstance(wc, (list, tuple)) and len(wc) == 3:
            vals = [_num(x) for x in wc]
            if all(v is not None for v in vals):
                lo, hi = WB_CUSTOM_RANGE
                p.wb_custom = tuple(_clamp(v, lo, hi) for v in vals)  # type: ignore[assignment]
        return p

    @classmethod
    def coerce(cls, params: "DevParams | Mapping[str, Any] | None") -> "DevParams":
        if isinstance(params, DevParams):
            return cls.from_dict(params.to_dict())  # re-validate (fields may have been mutated)
        return cls.from_dict(params)

    # ------------------------------------------------------------- serialize
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": DEV_VERSION,
            "exposure_ev": float(self.exposure_ev),
            "wb_mode": self.wb_mode,
            "wb_custom": [float(v) for v in self.wb_custom],
            "temperature_shift": float(self.temperature_shift),
            "tint_shift": float(self.tint_shift),
            "contrast": float(self.contrast),
            "saturation": float(self.saturation),
            "highlights": float(self.highlights),
            "shadows": float(self.shadows),
            "gamma_mode": self.gamma_mode,
            "gamma": float(self.gamma),
            "demosaic": self.demosaic,
        }
