"""Linear colour operations: camera calibration, saturation matrix, defringe.

Working space of the linear stages is linear Rec.709/sRGB (what ``rgb_cam``
produces). Calibration is defined on the **ProPhoto (ROMM) primaries**, as
requested by the engine spec: each primary is rotated around the grey axis
(hue, ±1 → ±25°) and its chroma scaled (saturation, ±1 → ×(1 ± 0.6)), then the
columns are rescaled so white stays white. The result is conjugated into the
Rec.709 working space and fused with ``rgb_cam`` into a single 3×3 matrix, so
it costs nothing per pixel.
"""

from __future__ import annotations

import numpy as np

from .common import F32, REC709, LowMap, block_mean, run_chunks, smoothstep

__all__ = ["saturation_matrix", "calibration_matrix", "defringe", "rgb_hue"]

# linear sRGB (D65) → XYZ and ProPhoto (D50) → XYZ, Bradford D50 → D65
_SRGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
_PP_XYZ50 = np.array([[0.7976749, 0.1351917, 0.0313534], [0.2880402, 0.7118741, 0.0000857],
                      [0.0, 0.0, 0.8252100]])
_BRAD_50_65 = np.array([[0.9555766, -0.0230393, 0.0631636], [-0.0282895, 1.0099416, 0.0210077],
                        [0.0122982, -0.0204830, 1.3299098]])
PP_TO_SRGB = np.linalg.inv(_SRGB_XYZ) @ _BRAD_50_65 @ _PP_XYZ50
SRGB_TO_PP = np.linalg.inv(PP_TO_SRGB)
CAL_HUE_DEG = 25.0
CAL_SAT = 0.6


def saturation_matrix(s: float) -> np.ndarray:
    """Rec.709-luminance-preserving saturation (DEV v1 semantics)."""
    return (s * np.eye(3) + (1.0 - s) * np.outer(np.ones(3), REC709)).astype(np.float64)


def _rot_grey(theta: float) -> np.ndarray:
    k = np.ones(3) / np.sqrt(3.0)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def calibration_matrix(rh: float, rs: float, gh: float, gs: float, bh: float, bs: float) -> np.ndarray:
    """3×3 matrix (linear Rec.709 → linear Rec.709) for the calibration sliders."""
    if not any((rh, rs, gh, gs, bh, bs)):
        return np.eye(3)
    k = np.ones(3) / np.sqrt(3.0)
    cols = []
    for i, (h, s) in enumerate(((rh, rs), (gh, gs), (bh, bs))):
        e = np.eye(3)[:, i]
        g = (e @ k) * k
        c = e - g
        c = (1.0 + CAL_SAT * s) * (_rot_grey(np.radians(CAL_HUE_DEG * h)) @ c)
        cols.append(g + c)
    m = np.stack(cols, axis=1)
    scale = np.linalg.solve(m, np.ones(3))  # keep white: M · 1 = 1
    m = m * scale[None, :]
    return PP_TO_SRGB @ m @ SRGB_TO_PP


def rgb_hue(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hexcone hue in [0, 6) (×60° = degrees), chroma C = max − min and max."""
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    c = mx - mn
    inv = F32(1.0) / np.maximum(c, F32(1e-12))
    h = (r - g) * inv
    h += F32(4.0)
    hg = (b - r) * inv
    hg += F32(2.0)
    np.copyto(h, hg, where=(mx == g))
    hr = (g - b) * inv
    hr += F32(6.0) * (hr < 0)
    np.copyto(h, hr, where=(mx == r))
    return h, c, mx


def defringe(lin: np.ndarray, amount: float, scale: float) -> None:
    """Removes purple / green fringes along high-contrast edges (in place, linear RGB).

    Edge map: gradient of √Y on a half-resolution grid, dilated (3×3 max) and
    slightly blurred. Pixels near strong edges whose hue falls in the purple–
    magenta (≈ 255–330°) or green (≈ 75–160°) ranges are desaturated towards
    their luminance, proportionally to ``amount`` and to the edge strength.
    """
    if amount <= 0:
        return
    from scipy import ndimage

    H, W, _ = lin.shape
    f = 2 if min(H, W) >= 16 else 1
    low = block_mean(lin, f)
    y = np.sqrt(np.clip(low @ REC709, 0.0, 4.0)).astype(np.float32)
    gy = np.abs(np.diff(y, axis=0, append=y[-1:]))
    gx = np.abs(np.diff(y, axis=1, append=y[:, -1:]))
    grad = ndimage.maximum_filter(np.maximum(gx, gy), size=3)
    edge = smoothstep(0.02 * f, 0.08 * f, grad.astype(np.float32))
    emap = LowMap(edge, H, W, f).full().reshape(-1)
    sel = np.flatnonzero(emap > 1e-3)  # only pixels near strong edges are touched
    if sel.size == 0:
        return
    flat = lin.reshape(-1, 3)
    a = F32(min(1.0, amount))

    def run(s: int, e: int) -> None:
        ix = sel[s:e]
        v = flat[ix]
        r, g, b = v[:, 0], v[:, 1], v[:, 2]
        h, c, mx = rgb_hue(r, g, b)
        h *= F32(60.0)
        purple = smoothstep(240.0, 265.0, h) * (F32(1.0) - smoothstep(320.0, 345.0, h))
        green = smoothstep(65.0, 85.0, h) * (F32(1.0) - smoothstep(150.0, 170.0, h))
        k = np.maximum(purple, green)
        k *= smoothstep(0.05, 0.25, c / np.maximum(mx, F32(1e-6)))
        k *= emap[ix]
        k *= a
        L = r * REC709[0] + g * REC709[1] + b * REC709[2]
        L *= k
        k = F32(1.0) - k
        v *= k[:, None]
        v += L[:, None]
        flat[ix] = v

    run_chunks(run, sel.size)
