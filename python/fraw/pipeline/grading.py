"""Colour grading (shadows / midtones / highlights / global wheels) in the encoded domain.

Zone weights depend only on the encoded Rec.709 luminance L:

* ``pivot = 0.5 − 0.25·balance`` (+balance → larger highlight zone);
* shadows ``1 − ss(Ls ± hw)`` around ``Ls = 0.6·pivot``, highlights ``ss(Lh ± hw)``
  around ``Lh = pivot + 0.4·(1 − pivot)``, midtones the remainder; the
  transition half width ``hw = 0.05 + 0.25·blending``.

Each wheel adds ``w(L)·sat·0.18·c(hue)`` (``c`` = zero-luminance unit chroma of
the hue at full HSV saturation) and ``w(L)·lum·0.2`` (luminance shift). The
chroma tint is faded in over the deepest blacks so black stays black. The total
offset is a function of L only → a 1024-entry table, one lookup per pixel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import F32, REC709, run_chunks

__all__ = ["grading_table", "apply_grading", "TABLE"]

TABLE = 1024
SAT_K = 0.18
LUM_K = 0.2


def _hue_chroma(deg: float) -> np.ndarray:
    h = (deg % 360.0) / 60.0
    x = 1 - abs(h % 2 - 1)
    rgb = [(1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x)][int(h) % 6]
    v = np.array(rgb, dtype=np.float64)
    v = v - float(v @ REC709)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _ss(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def grading_table(p: Any) -> np.ndarray | None:
    zones = ("shadows", "midtones", "highlights", "global")
    if not any(float(getattr(p, f"grade_{z}_sat")) or float(getattr(p, f"grade_{z}_lum")) for z in zones):
        return None
    L = (np.arange(TABLE) + 0.5) / TABLE
    pivot = 0.5 - 0.25 * float(p.grade_balance)
    hw = 0.05 + 0.25 * float(p.grade_blending)
    ls, lh = 0.6 * pivot, pivot + 0.4 * (1 - pivot)
    ws = 1 - _ss(ls - hw, ls + hw, L)
    wh = _ss(lh - hw, lh + hw, L)
    wm = np.clip(1 - ws - wh, 0, 1)
    weights = {"shadows": ws, "midtones": wm, "highlights": wh, "global": np.ones_like(L)}
    fade = 0.15 + 0.85 * _ss(0.0, 0.12, L)
    off = np.zeros((TABLE, 3))
    for z in zones:
        sat, lum = float(getattr(p, f"grade_{z}_sat")), float(getattr(p, f"grade_{z}_lum"))
        w = weights[z]
        if sat:
            off += (w * fade * sat * SAT_K)[:, None] * _hue_chroma(float(getattr(p, f"grade_{z}_hue")))[None, :]
        if lum:
            off += (w * lum * LUM_K)[:, None]
    return off.astype(np.float32)


def apply_grading(flat: np.ndarray, table: np.ndarray | None) -> None:
    if table is None:
        return
    k = F32(TABLE)

    def run(s: int, e: int) -> None:
        v = flat[s:e]
        L = v @ REC709
        L *= k
        np.maximum(L, F32(0.0), out=L)
        np.minimum(L, F32(TABLE - 1), out=L)
        idx = L.astype(np.int32)
        v += table[idx]

    run_chunks(run, flat.shape[0])
