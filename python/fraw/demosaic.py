"""Bayer demosaicing — bilinear and AHD (Hirakawa & Parks), vectorized numpy.

Both functions take a normalized single-channel CFA image (float, H×W) and the
2×2 ``cfa_pattern`` (row-major from (0,0), ``0=R 1=G 2=B``, SPEC §4.1) and
return an H×W×3 float32 RGB image (camera space, not white-balanced here).
Known samples are returned *exactly* in their own channel.

Border handling: samples outside the image are taken by mirror reflection
without repeating the edge sample (``numpy.pad(mode="reflect")`` /
``scipy.ndimage`` ``mode="mirror"``). That reflection preserves the CFA parity,
so a reflected sample always has the colour the pattern predicts.

AHD only applies to the four Bayer layouts (RGGB, BGGR, GRBG, GBRG); they are
normalized to RGGB by shifting the origin one row and/or column through the
reflected padding, and the shift is undone when cropping the result. Any other
(non-Bayer) 2×2 layout falls back to the generic bilinear interpolation.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import numpy as np
from scipy import ndimage

__all__ = ["bilinear", "ahd", "bayer_offset", "mosaic", "METHODS", "demosaic"]

# Separable [1 2 1] ⊗ [1 2 1] kernel; on a Bayer mask its response is always
# 4 for R and B and 8 for G, which yields classic bilinear interpolation.
_K121 = np.array([1.0, 2.0, 1.0], dtype=np.float32)

# sRGB(D65) → XYZ, used only to measure homogeneity in CIELab.
_RGB2XYZ = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]],
    dtype=np.float32,
)
_D65 = np.array([0.950456, 1.0, 1.088754], dtype=np.float32)

_PAD = 8  # even (keeps CFA parity); > receptive field of the AHD core (6 px)
_STRIP = 256  # AHD rows per strip (bounded memory, multithreaded)


def _check(cfa: np.ndarray, pattern: Sequence[int]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    a = np.asarray(cfa)
    if a.ndim != 2:
        raise ValueError("cfa must be a 2-D array")
    if a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError("cfa must be at least 2x2")
    pat = tuple(int(v) for v in pattern)
    if len(pat) != 4 or any(v not in (0, 1, 2) for v in pat) or not {0, 1, 2} <= set(pat):
        raise ValueError("pattern must be 4 values in {0,1,2} containing R, G and B")
    return a.astype(np.float32, copy=False), pat  # type: ignore[return-value]


def bayer_offset(pattern: Sequence[int]) -> tuple[int, int] | None:
    """(dy, dx) of the red sample if ``pattern`` is a Bayer layout, else None."""
    pat = tuple(int(v) for v in pattern)
    if sorted(pat) != [0, 1, 1, 2]:
        return None
    r = pat.index(0)
    if pat[3 - r] != 2:  # blue must be on the red's diagonal (greens on the other)
        return None
    return r >> 1, r & 1


def _masks(shape: tuple[int, int], pat: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Boolean masks (R, G, B) for an image of ``shape`` with pattern ``pat``."""
    h, w = shape
    idx = ((np.arange(h)[:, None] & 1) << 1) | (np.arange(w)[None, :] & 1)
    lut = np.array(pat, dtype=np.uint8)[idx]
    return [lut == c for c in range(3)]


def mosaic(rgb: np.ndarray, pattern: Sequence[int]) -> np.ndarray:
    """Samples an H×W×3 image through the CFA (test / simulation helper)."""
    rgb = np.asarray(rgb)
    pat = tuple(int(v) for v in pattern)
    out = np.empty(rgb.shape[:2], dtype=np.float32)
    for i in range(4):
        out[i >> 1 :: 2, i & 1 :: 2] = rgb[i >> 1 :: 2, i & 1 :: 2, pat[i]]
    return out


# ---------------------------------------------------------------------------
# Bilinear
# ---------------------------------------------------------------------------
def _conv121(a: np.ndarray) -> np.ndarray:
    t = ndimage.correlate1d(a, _K121, axis=1, mode="mirror")
    return ndimage.correlate1d(t, _K121, axis=0, mode="mirror")


def bilinear(cfa: np.ndarray, pattern: Sequence[int]) -> np.ndarray:
    """Classic bilinear demosaic as normalized convolution (works for any 2×2 layout).

    For each colour c: ``out_c = K*(x·m_c) / K*m_c`` with ``K = [1 2 1]ᵀ[1 2 1]``.
    On a Bayer mosaic this is exactly the textbook average of the 2 or 4 nearest
    samples of the missing colour.
    """
    x, pat = _check(cfa, pattern)
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for c, m in enumerate(_masks(x.shape, pat)):
        mf = m.astype(np.float32)
        num = _conv121(x * mf)
        den = _conv121(mf)
        ch = num / den
        ch[m] = x[m]  # known samples, bit-exact
        out[..., c] = ch
    return out


# ---------------------------------------------------------------------------
# AHD
# ---------------------------------------------------------------------------
def _s(a: np.ndarray, oy: int, ox: int) -> np.ndarray:
    """``out[y, x] = a[y + oy, x + ox]`` (wraps; callers discard the margin)."""
    return np.roll(a, (-oy, -ox), axis=(0, 1))


def _diff_interp(d: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a colour difference known only at R (or B) sites of RGGB."""
    t = _s(d, 0, -1) + 2.0 * d + _s(d, 0, 1)
    u = _s(t, -1, 0) + 2.0 * t + _s(t, 1, 0)
    return u * np.float32(0.25)  # K-response of an R/B mask is 4 everywhere


def _lab(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = _RGB2XYZ / _D65[:, None]
    xyz = [m[i, 0] * r + m[i, 1] * g + m[i, 2] * b for i in range(3)]
    f = []
    for t in xyz:
        f.append(np.where(t > 0.008856, np.cbrt(np.maximum(t, 0.008856)), 7.787 * t + 16.0 / 116.0).astype(np.float32))
    fx, fy, fz = f
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


_N4 = ((0, -1), (0, 1), (-1, 0), (1, 0))  # left, right, up, down


def _ahd_core(x: np.ndarray, rmask: np.ndarray, bmask: np.ndarray, gmask: np.ndarray) -> np.ndarray:
    """AHD on an RGGB-aligned padded tile. Margin of ``_PAD`` px is garbage."""
    # 1. Green, Hamilton–Adams in each direction, limited to the neighbour range.
    gl, gr = _s(x, 0, -1), _s(x, 0, 1)
    gh = 0.5 * (gl + gr) + 0.25 * (2.0 * x - _s(x, 0, -2) - _s(x, 0, 2))
    gh = np.clip(gh, np.minimum(gl, gr), np.maximum(gl, gr))
    gu, gd = _s(x, -1, 0), _s(x, 1, 0)
    gv = 0.5 * (gu + gd) + 0.25 * (2.0 * x - _s(x, -2, 0) - _s(x, 2, 0))
    gv = np.clip(gv, np.minimum(gu, gd), np.maximum(gu, gd))

    rgbs = []
    labs = []
    for gi in (gh, gv):
        g = np.where(gmask, x, gi)
        # 2. R/B from the colour differences of this direction.
        r = g + _diff_interp(np.where(rmask, x - g, np.float32(0)))
        b = g + _diff_interp(np.where(bmask, x - g, np.float32(0)))
        rgbs.append((r, g, b))
        labs.append(_lab(r, g, b))

    # 3. Homogeneity: neighbours whose Lab distance is within the adaptive ε.
    ld, cd = [], []
    for L, A, B in labs:
        ld.append([np.abs(L - _s(L, *o)) for o in _N4])
        cd.append([(A - _s(A, *o)) ** 2 + (B - _s(B, *o)) ** 2 for o in _N4])
    leps = np.minimum(np.maximum(ld[0][0], ld[0][1]), np.maximum(ld[1][2], ld[1][3]))
    ceps = np.minimum(np.maximum(cd[0][0], cd[0][1]), np.maximum(cd[1][2], cd[1][3]))
    homo = []
    for d in range(2):
        h = np.zeros(x.shape, dtype=np.float32)
        for i in range(4):
            h += (ld[d][i] <= leps) & (cd[d][i] <= ceps)
        # 4. Smoothed homogeneity map: 3×3 box sum.
        t = _s(h, 0, -1) + h + _s(h, 0, 1)
        homo.append(_s(t, -1, 0) + t + _s(t, 1, 0))

    # 5. Pick the more homogeneous direction; average on ties.
    hh, hv = homo
    wh = np.where(hh > hv, np.float32(1), np.where(hh < hv, np.float32(0), np.float32(0.5)))
    out = np.empty(x.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = rgbs[1][c] + wh * (rgbs[0][c] - rgbs[1][c])
    return out


def ahd(cfa: np.ndarray, pattern: Sequence[int], *, threads: int | None = None) -> np.ndarray:
    """Adaptive Homogeneity-Directed demosaic (Hirakawa & Parks 2005).

    Processed in horizontal strips (bounded memory, multithreaded; numpy
    releases the GIL). Strips overlap by ``_PAD`` rows so the result is
    identical to processing the whole image at once.
    """
    x, pat = _check(cfa, pattern)
    off = bayer_offset(pat)
    if off is None:
        return bilinear(x, pat)
    dy, dx = off
    h, w = x.shape
    P = _PAD
    xp = np.pad(x, ((P + dy, P), (P + dx, P)), mode="reflect")  # (P+dy, P+dx) → R at (0, 0)
    out = np.empty((h, w, 3), dtype=np.float32)
    wp = xp.shape[1]
    col_odd = (np.arange(wp) & 1).astype(bool)[None, :]

    def run(y0: int) -> None:
        y1 = min(y0 + _STRIP, h)
        # rows y0..y1 of the output live at padded rows y0+P+dy..y1+P+dy; start even.
        top = y0 + dy  # = (y0 + P + dy) - P, even because y0 and P are even... dy fixes parity below
        top -= top & 1
        bot = min(y1 + P + dy + P, xp.shape[0])
        tile = xp[top:bot]
        row_odd = (np.arange(top, bot) & 1).astype(bool)[:, None]
        rmask = ~row_odd & ~col_odd
        bmask = row_odd & col_odd
        gmask = row_odd ^ col_odd
        res = _ahd_core(tile, rmask, bmask, gmask)
        r0 = y0 + P + dy - top
        out[y0:y1] = res[r0 : r0 + (y1 - y0), P + dx : P + dx + w]

    starts = list(range(0, h, _STRIP))
    n = threads if threads is not None else min(len(starts), os.cpu_count() or 1)
    if n <= 1 or len(starts) == 1:
        for y0 in starts:
            run(y0)
    else:
        with ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(run, starts))

    # Known samples, bit-exact.
    for i in range(4):
        out[i >> 1 :: 2, i & 1 :: 2, pat[i]] = x[i >> 1 :: 2, i & 1 :: 2]
    return out


METHODS = {"bilinear": bilinear, "ahd": ahd}


def demosaic(cfa: np.ndarray, pattern: Sequence[int], method: str = "ahd") -> np.ndarray:
    try:
        fn = METHODS[method]
    except KeyError:
        raise ValueError(f"unknown demosaic method {method!r}") from None
    return fn(cfa, pattern)
