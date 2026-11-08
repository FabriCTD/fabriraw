"""Automatic tone from luminance percentiles (simulated with the real tone stage)."""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["auto_tone"]

TARGET_MEDIAN = 0.18
CLIP_MAX = 0.005


def _q(a: np.ndarray, q: float) -> float:
    return float(np.quantile(a, q)) if a.size else 0.0


def auto_tone(y0: np.ndarray, render: Callable[[dict], np.ndarray], mean_sat: float | None = None) -> dict:
    """Robust auto tone.

    ``y0``: linear scene luminance at exposure 0 (WB + colour matrix applied,
    no tone controls). ``render(params)`` returns the linear RGB (N×3) after the
    tone stage for a dict with exposure_ev/highlights/shadows/whites/blacks.

    Exposure puts the median at ~0.18; highlights (then, if needed, a smaller
    exposure) keep the 99.5th percentile below white so ≤ 0.5 % clips; shadows
    lift very deep 5th percentiles; whites/blacks stretch the end points;
    contrast follows the 5–95 % dynamic range.
    """
    y = np.asarray(y0, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    y = np.maximum(y, 0.0)
    if y.size == 0:
        return {"exposure_ev": 0.0, "contrast": 0.0, "highlights": 0.0, "shadows": 0.0, "whites": 0.0, "blacks": 0.0}
    med = max(_q(y, 0.5), 1e-5)
    ev = float(np.clip(np.log2(TARGET_MEDIAN / med), -4.0, 4.0))
    sc = y * 2.0**ev
    p005, p05, p95, p995 = (_q(sc, q) for q in (0.005, 0.05, 0.95, 0.995))
    hl = 0.0
    if p995 > 0.85:
        hl = -float(np.clip(np.log2(p995 / 0.85) / 1.0, 0.0, 1.0))
    sh = float(np.clip(np.log2(0.02 / max(p05, 1e-6)) / 4.0, 0.0, 0.6)) if p05 < 0.02 else 0.0
    wh = float(np.clip((0.7 - p995) * 0.8, 0.0, 0.4)) if p995 < 0.7 else 0.0
    bl = -float(np.clip(np.log2(p005 / 0.005) / 6.0, 0.0, 0.4)) if p005 > 0.01 else 0.0
    dr = float(np.log2(max(p95, 1e-6) / max(p05, 1e-6)))
    if dr < 4.5:
        ct = float(np.clip((4.5 - dr) * 0.08, 0.0, 0.35))
    elif dr > 8.0:
        ct = -float(np.clip((dr - 8.0) * 0.05, 0.0, 0.25))
    else:
        ct = 0.0
    prm = {"exposure_ev": ev, "highlights": hl, "shadows": sh, "whites": wh, "blacks": bl}
    ev_cap = 4.0
    for _ in range(14):
        lin = np.asarray(render(prm), dtype=np.float64).reshape(-1, 3)
        lin = lin[np.all(np.isfinite(lin), axis=1)]
        out = lin @ np.array([0.2126, 0.7152, 0.0722])
        clip = float(np.mean(lin.max(axis=1) >= 0.999)) if lin.size else 0.0
        m = max(_q(out, 0.5), 1e-6)
        corr = float(np.clip(np.log2(TARGET_MEDIAN / m), -1.0, 1.0))
        if clip > CLIP_MAX:
            if prm["whites"] > 0:
                prm["whites"] = max(0.0, prm["whites"] - 0.2)
                continue
            if prm["highlights"] > -1.0:
                prm["highlights"] = max(-1.0, prm["highlights"] - 0.25)
                continue
            if prm["whites"] > -0.6:
                prm["whites"] = max(-0.6, prm["whites"] - 0.2)
                continue
            # only exposure can avoid the clipping: cap it (the median may stay lower)
            ev_cap = prm["exposure_ev"] - 0.2
            corr = min(corr, 0.0)
        new_ev = float(np.clip(prm["exposure_ev"] + 0.9 * corr, -4.0, min(4.0, ev_cap)))
        if abs(new_ev - prm["exposure_ev"]) < 0.01 and clip <= CLIP_MAX:
            break
        prm["exposure_ev"] = new_ev
    res = {
        "exposure_ev": round(prm["exposure_ev"], 2),
        "contrast": round(ct, 2),
        "highlights": round(prm["highlights"], 2),
        "shadows": round(prm["shadows"], 2),
        "whites": round(prm["whites"], 2),
        "blacks": round(prm["blacks"], 2),
    }
    if mean_sat is not None and np.isfinite(mean_sat):
        res["vibrance"] = round(float(np.clip((0.3 - mean_sat) * 1.2, 0.0, 0.35)), 2)
    return res
