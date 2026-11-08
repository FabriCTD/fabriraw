"""8-band HSL mixer + vibrance, in the output-encoded (perceptual) domain.

Hue is the standard HSV hexcone hue, so the band centres are those of
``devparams.HSL_BAND_HUES`` (red 0°, orange 30°, yellow 60°, green 120°, aqua
180°, blue 225°, purple 270°, magenta 315°). Per-band values are turned into
three smooth functions of hue (1024-entry tables): between two neighbouring
band centres the weight goes from one band to the next with a smoothstep, so
the eight weights form a smooth circular partition of unity.

Per pixel (weights scaled by a chroma gate ``min(1, 8·S)`` so greys are untouched):

* hue: rotation around the grey axis (Rodrigues), ±1 → ±30°; preserves the
  RGB mean and the chroma magnitude;
* saturation: ``L + (rgb − L)·(1 + s)`` (Rec.709 L of the encoded values);
* luminance: ``rgb·(1 + 0.45·l)`` — scales the colour, keeping hue and
  relative chroma;
* vibrance (positive): ``+vib·(1 − S)²`` saturation, reduced by 60 % on skin
  hues (≈ 0–60°, hue table); negative vibrance: ``vib·(1 − S/2)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..devparams import HSL_BAND_HUES, HSL_BANDS
from .color import rgb_hue
from .common import F32, REC709, run_chunks

__all__ = ["hsl_tables", "apply_hsl", "HUE_SHIFT_DEG", "TABLE"]

HUE_SHIFT_DEG = 30.0
LUM_K = 0.45
TABLE = 1024


def _band_table(values: list[float]) -> np.ndarray:
    centers = np.array([HSL_BAND_HUES[b] for b in HSL_BANDS], dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    h = (np.arange(TABLE) + 0.5) * 360.0 / TABLE
    out = np.empty(TABLE)
    n = len(centers)
    for i in range(n):
        c0, c1 = centers[i], centers[(i + 1) % n] + (360.0 if i == n - 1 else 0.0)
        hh = np.where(h < centers[0], h + 360.0, h)
        sel = (hh >= c0) & (hh < c1)
        t = (hh[sel] - c0) / (c1 - c0)
        t = t * t * (3 - 2 * t)
        out[sel] = vals[i] * (1 - t) + vals[(i + 1) % n] * t
    return out.astype(np.float32)


def _skin_table() -> np.ndarray:
    h = (np.arange(TABLE) + 0.5) * 360.0 / TABLE
    def ss(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    skin = ss(0.0, 15.0, h) * (1 - ss(40.0, 60.0, h))
    return (1.0 - 0.6 * skin).astype(np.float32)


_SKIN = _skin_table()


def hsl_tables(p: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    hv = [float(getattr(p, f"hsl_{b}_h")) for b in HSL_BANDS]
    sv = [float(getattr(p, f"hsl_{b}_s")) for b in HSL_BANDS]
    lv = [float(getattr(p, f"hsl_{b}_l")) for b in HSL_BANDS]
    if not any(hv) and not any(sv) and not any(lv):
        return None
    return (
        _band_table(hv) * F32(np.radians(HUE_SHIFT_DEG)) if any(hv) else None,
        _band_table(sv) if any(sv) else None,
        _band_table(lv) * F32(LUM_K) if any(lv) else None,
    )


def apply_hsl(flat: np.ndarray, tables, vibrance: float) -> None:
    """In place on an N×3 encoded RGB array."""
    if tables is None and not vibrance:
        return
    th, ts, tl = tables if tables is not None else (None, None, None)
    vib = F32(vibrance)
    inv3 = F32(1.0 / np.sqrt(3.0))
    tscale = F32(TABLE / 6.0)
    tmax = F32(TABLE - 1)

    def run(s: int, e: int) -> None:
        v = flat[s:e]
        r, g, b = v[:, 0], v[:, 1], v[:, 2]
        h, c, mx = rgb_hue(r, g, b)
        sat = c / np.maximum(mx, F32(1e-6))
        np.minimum(sat, F32(1.0), out=sat)
        gate = sat * F32(8.0)
        np.minimum(gate, F32(1.0), out=gate)
        h *= tscale
        np.minimum(h, tmax, out=h)
        idx = h.astype(np.int32)
        ds = None
        if ts is not None:
            ds = ts[idx]
            ds *= gate
        if vib:
            u = F32(1.0) - sat
            if vib > 0:
                u *= u
                u *= _SKIN[idx]
                u *= vib
            else:
                u = F32(0.5) + F32(0.5) * u
                u *= vib
            ds = u if ds is None else ds + u
        if th is not None:
            ang = th[idx]
            ang *= gate
            a2 = ang * ang
            cs = F32(1.0) - a2 * (F32(0.5) - a2 * F32(1.0 / 24.0))
            sn = ang * (F32(1.0) - a2 * F32(1.0 / 6.0))
            sn *= inv3
            m = r + g
            m += b
            m *= F32(1.0 / 3.0)
            nr = (r - m) * cs + (b - g) * sn
            ng = (g - m) * cs + (r - b) * sn
            nb = (b - m) * cs + (g - r) * sn
            nr += m
            ng += m
            nb += m
            v[:, 0] = nr
            v[:, 1] = ng
            v[:, 2] = nb
        if ds is not None:
            L = v @ REC709
            ds += F32(1.0)
            np.maximum(ds, F32(0.0), out=ds)
            k = ds[:, None]
            v -= L[:, None]
            v *= k
            v += L[:, None]
        if tl is not None:
            dl = tl[idx]
            dl *= gate
            dl += F32(1.0)
            v *= dl[:, None]

    run_chunks(run, flat.shape[0])
