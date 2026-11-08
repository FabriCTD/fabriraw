"""Post-crop vignette and film grain (output-encoded domain).

Both are defined in **final-frame coordinates** (after geometry and crop), so
a zoomed region, the preview and the export agree.

Vignette: normalized coords ``u, v ∈ [−1, 1]``; distance is a superellipse
``(|u|ⁿ + |v|ⁿ)^(1/n)`` with ``n = 2 + 6·max(0, −roundness)`` (towards a
rounded rectangle) and, for positive roundness, ``v`` rescaled towards a circle
in pixel space. Mask ``ss(m − w/2, m + w/2, d)`` with ``m = 0.4 + 0.8·midpoint``
and ``w = 0.05 + feather``. Negative amounts darken multiplicatively
(``2**(1.8·amount·mask)``, −1.8 EV in the corners at −1), sparing bright pixels
by ``vignette_highlights``; positive amounts lighten towards white.

Grain: value noise on a lattice of ``1 + 3·size`` full-res pixels (at least
one output pixel), hashed from the integer lattice coordinates and a per-image
seed → deterministic and position-stable. ``roughness`` mixes in an octave at
half the cell size. Amplitude ``0.09·amount`` (encoded units) weighted by
``0.3 + 2.8·L·(1 − L)`` (strongest in the midtones), monochromatic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import F32, REC709, run_chunks, smoothstep

__all__ = ["vignette_mask", "apply_vignette", "grain_field", "apply_grain"]


def vignette_mask(p: Any, frame_w: float, frame_h: float, x0: float, y0: float, w: int, h: int,
                  s_out: float) -> np.ndarray:
    xs = (x0 + (np.arange(w) + 0.5) / s_out) / frame_w * 2.0 - 1.0
    ys = (y0 + (np.arange(h) + 0.5) / s_out) / frame_h * 2.0 - 1.0
    rnd = float(p.vignette_roundness)
    u = np.abs(xs)[None, :].astype(np.float32)
    v = np.abs(ys)[:, None].astype(np.float32)
    if rnd > 0:  # towards a circle in pixel space
        asp = frame_h / frame_w
        if asp < 1:
            v = v * F32(1.0 + rnd * (asp - 1.0))
        else:
            u = u * F32(1.0 + rnd * (1.0 / asp - 1.0))
    n = 2.0 + 6.0 * max(0.0, -rnd)
    if n == 2.0:
        d = np.sqrt(u * u + v * v)
    else:
        d = np.power(np.power(u, F32(n)) + np.power(v, F32(n)), F32(1.0 / n))
    m = 0.4 + 0.8 * float(p.vignette_midpoint)
    fw = 0.05 + float(p.vignette_feather)
    return smoothstep(m - fw / 2, m + fw / 2, d).astype(np.float32)


def apply_vignette(img: np.ndarray, mask: np.ndarray, amount: float, protect: float) -> None:
    """Negative: ×2**(1.8·amount·mask) (−1 → −1.8 EV in the corners), sparing bright pixels by
    ``protect``; positive: towards white, ``rgb + amount·mask·(1 − rgb)``."""
    flat = img.reshape(-1, 3)
    mk = mask.reshape(-1)
    a = F32(amount)
    pr = F32(protect)
    if a < 0:
        gain = np.exp2(F32(1.8) * a * mk).astype(np.float32)

    def run(s: int, e: int) -> None:
        v = flat[s:e]
        if a < 0:
            g = gain[s:e]
            if pr > 0:
                L = v @ REC709
                w = smoothstep(0.5, 1.0, L)
                w *= pr
                g = g + (F32(1.0) - g) * w
            v *= g[:, None]
        else:
            m = mk[s:e] * a
            v += m[:, None] * (F32(1.0) - v)

    run_chunks(run, flat.shape[0])


def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        h = (ix.astype(np.uint32) * np.uint32(0x8DA6B343)) ^ (iy.astype(np.uint32) * np.uint32(0xD8163841))
        h ^= np.uint32(seed & 0xFFFFFFFF) * np.uint32(0xCB1AB31F)
        h ^= h >> np.uint32(13)
        h *= np.uint32(0x5BD1E995)
        h ^= h >> np.uint32(15)
        h2 = h * np.uint32(0x27D4EB2D)
        h2 ^= h2 >> np.uint32(16)
    a = (h & np.uint32(0xFFFF)).astype(np.float32) * F32(1.0 / 65535.0)
    b = (h2 & np.uint32(0xFFFF)).astype(np.float32) * F32(1.0 / 65535.0)
    return (a + b - F32(1.0)) * F32(1.73)  # triangular, unit-ish variance


def _octave(cell: float, x0: float, y0: float, w: int, h: int, s_out: float, seed: int) -> np.ndarray:
    # lattice indices covering the output; bilinear interpolation as sparse matrices
    xs = (x0 + (np.arange(w) + 0.5) / s_out) / cell - 0.5
    ys = (y0 + (np.arange(h) + 0.5) / s_out) / cell - 0.5
    ix0, iy0 = int(np.floor(xs[0])), int(np.floor(ys[0]))
    nx, ny = int(np.floor(xs[-1])) - ix0 + 2, int(np.floor(ys[-1])) - iy0 + 2
    gx, gy = np.meshgrid(np.arange(ix0, ix0 + nx), np.arange(iy0, iy0 + ny))
    lat = _hash(gx, gy, seed)
    from scipy import sparse

    def interp(coords: np.ndarray, origin: int, n: int) -> sparse.csr_matrix:
        f = np.floor(coords)
        t = (coords - f).astype(np.float32)
        t = t * t * (3 - 2 * t)
        i = (f - origin).astype(np.int64)
        rows = np.repeat(np.arange(coords.size), 2)
        cols = np.stack([i, np.minimum(i + 1, n - 1)], axis=1).ravel()
        vals = np.stack([1 - t, t], axis=1).ravel()
        return sparse.csr_matrix((vals, (rows, cols)), shape=(coords.size, n))

    ry = interp(ys, iy0, ny)
    rx = interp(xs, ix0, nx)
    t = np.asarray(ry @ lat, dtype=np.float32)  # h × nx
    return np.ascontiguousarray(np.asarray(rx @ t.T, dtype=np.float32).T)


def grain_field(p: Any, seed: int, x0: float, y0: float, w: int, h: int, s_out: float) -> np.ndarray:
    size = float(p.grain_size)
    rough = float(p.grain_roughness)
    cell = max(1.0 + 3.0 * size, 1.0 / s_out)  # full-res px, ≥ 1 output px
    n = _octave(cell, x0, y0, w, h, s_out, seed)
    if rough > 0:
        cell2 = max(cell * 0.5, 1.0 / s_out)
        n2 = _octave(cell2, x0, y0, w, h, s_out, seed ^ 0x5F3759DF)
        n = n * F32(1.0 - 0.5 * rough) + n2 * F32(0.75 * rough)
    return n


def apply_grain(img: np.ndarray, field: np.ndarray, amount: float) -> None:
    flat = img.reshape(-1, 3)
    fl = field.reshape(-1)
    a = F32(0.09 * amount)

    def run(s: int, e: int) -> None:
        v = flat[s:e]
        L = v @ REC709
        np.maximum(L, F32(0.0), out=L)
        np.minimum(L, F32(1.0), out=L)
        k = (F32(1.0) - L) * L
        k *= F32(2.8)
        k += F32(0.3)
        k *= fl[s:e]
        k *= a
        v += k[:, None]

    run_chunks(run, flat.shape[0])
