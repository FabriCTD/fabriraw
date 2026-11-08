"""Shared numeric helpers for the development pipeline (float32, numpy/scipy only).

Performance model: numpy pays a fixed cost per call and is memory bound, so
per-pixel work is written as *row-block kernels* (≈ 64 k pixels per block,
cache resident, run on a thread pool — numpy releases the GIL), and spatial
filters with large footprints run on **reduced grids** (integer block means)
whose results are bilinearly upsampled *inside* the kernels (:class:`LowMap`)
instead of materializing full-resolution maps.

Radii are given in pixels of the array being processed; callers convert from
full-resolution pixels with the current output scale so preview, export and
zoomed regions look alike.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Callable, Sequence

import numpy as np
from scipy import ndimage, sparse

__all__ = [
    "REC709", "F32", "executor", "run_chunks", "run_rows", "run_blocks", "smoothstep", "smoothstep_",
    "luminance", "resample_matrix", "resize", "gaussian_blur", "box_blur", "block_mean", "LowMap",
    "guided_coeffs", "guided_filter", "quantize8",
]

F32 = np.float32
REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_R, _G, _B = (F32(v) for v in REC709)

_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
_CHUNK = 65536  # pixels per work item


def executor() -> ThreadPoolExecutor | None:
    """Shared worker pool. Work items never submit to the pool themselves (no nesting)."""
    global _POOL
    n = min(8, os.cpu_count() or 1)
    if n <= 1:
        return None
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="fraw-pipe")
    return _POOL


def _run(fn: Callable[[int, int], None], bounds: list[tuple[int, int]]) -> None:
    ex = executor()
    if ex is None or len(bounds) <= 1 or threading.current_thread().name.startswith("fraw-pipe"):
        for s, e in bounds:
            fn(s, e)
    else:
        for f in [ex.submit(fn, s, e) for s, e in bounds]:
            f.result()


def run_chunks(fn: Callable[[int, int], None], n: int, chunk: int = _CHUNK) -> None:
    """Calls ``fn(start, end)`` over ``[0, n)`` in chunks, multithreaded."""
    _run(fn, [(s, min(s + chunk, n)) for s in range(0, n, chunk)])


def run_blocks(fn: Callable[[int, int], None], h: int, w: int, chunk: int = _CHUNK) -> None:
    """Calls ``fn(r0, r1)`` over row blocks of about ``chunk`` pixels, multithreaded."""
    step = max(1, chunk // max(1, w))
    _run(fn, [(s, min(s + step, h)) for s in range(0, h, step)])


def run_rows(fn: Callable[[int, int], None], n: int, parts: int = 16) -> None:
    step = max(1, -(-n // parts))
    _run(fn, [(s, min(s + step, n)) for s in range(0, n, step)])


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep (float32, ufunc-only — much cheaper than ``np.clip``)."""
    t = (x - F32(e0)) * F32(1.0 / (e1 - e0))
    np.maximum(t, F32(0.0), out=t)
    np.minimum(t, F32(1.0), out=t)
    return t * t * (F32(3.0) - F32(2.0) * t)


smoothstep_ = smoothstep


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of an H×W×3 (or N×3) float array."""
    return img[..., 0] * _R + img[..., 1] * _G + img[..., 2] * _B


def quantize8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] → uint8 (round half up), chunked."""
    a = np.ascontiguousarray(img, dtype=np.float32)
    flat = a.reshape(-1)
    out = np.empty(flat.shape, dtype=np.uint8)

    def k(s: int, e: int) -> None:
        v = flat[s:e] * F32(255.0)
        v += F32(0.5)
        np.maximum(v, F32(0.0), out=v)
        np.minimum(v, F32(255.0), out=v)
        out[s:e] = v

    run_chunks(k, flat.size, 3 * _CHUNK)
    return out.reshape(a.shape)


# ---------------------------------------------------------------------------
# Reduced grids
# ---------------------------------------------------------------------------
def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """Mean over f×f blocks (the remainder rows/cols are dropped). Strided sums: cheap."""
    f = int(f)
    a = np.asarray(a, dtype=np.float32)
    if f <= 1:
        return a.copy()
    H, W = a.shape[:2]
    if H < f or W < f:
        return a.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    h, w = H // f, W // f
    if f <= 4:
        o = a[0 : h * f : f, 0 : w * f : f].copy()
        for i in range(f):
            for j in range(f):
                if i or j:
                    o += a[i : h * f : f, j : w * f : f]
    else:
        t = a[0 : h * f : f].copy()
        for k in range(1, f):
            t += a[k : h * f : f]
        o = t[:, 0 : w * f : f].copy()
        for k in range(1, f):
            o += t[:, k : w * f : f]
    o *= F32(1.0 / (f * f))
    return o


class LowMap:
    """Bilinear upsampler of block-mean grids (cell k ↔ pixels [k·f, (k+1)·f)) to H×W, by row blocks.

    ``planes``: one (h, w) array or a sequence of them. The columns are expanded
    once at construction (h × W per plane, 1/f of the full cost); :meth:`rows`
    then only lerps two expanded rows per output row (contiguous, long loops).
    """

    def __init__(self, planes, H: int, W: int, f: int):
        f = max(1, int(f))
        self.single = isinstance(planes, np.ndarray) and planes.ndim == 2
        plist = [planes] if self.single else list(planes)
        self.f, self.H, self.W = f, H, W
        h = plist[0].shape[0]
        self.planes = [self._expand_cols(np.asarray(p, dtype=np.float32), W, f) for p in plist]
        u = (np.arange(H) + 0.5) / f - 0.5
        y0 = np.clip(np.floor(u), 0, h - 1).astype(np.intp)
        y1 = np.minimum(y0 + 1, h - 1)
        fy = np.clip(u - y0, 0.0, 1.0).astype(np.float32)
        fy[y1 == y0] = 0.0
        self.y0, self.y1, self.fy = y0, y1, fy[:, None]

    @staticmethod
    def _expand_cols(p: np.ndarray, W: int, f: int) -> np.ndarray:
        h, w = p.shape
        if f == 1:
            return np.ascontiguousarray(p[:, :W]) if w >= W else np.ascontiguousarray(
                np.concatenate([p] + [p[:, -1:]] * (W - w), axis=1))
        nb = -(-W // f)
        right = max(1, nb + 1 - w)
        q = np.concatenate([p[:, :1], p] + [p[:, -1:]] * right, axis=1)
        m = (np.arange(f) + 0.5) / f - 0.5
        off = np.floor(m).astype(int)
        t = (m - off).astype(np.float32)
        out = np.empty((h, nb * f), dtype=np.float32)
        l2 = q[:, 1 : nb + 1]
        dA = l2 - q[:, 0:nb]
        dB = q[:, 2 : nb + 2] - l2
        for k in range(f):
            out[:, k::f] = l2 + (dA * (t[k] - F32(1.0)) if off[k] < 0 else dB * t[k])
        return np.ascontiguousarray(out[:, :W])

    def _one(self, p: np.ndarray, r0: int, r1: int) -> np.ndarray:
        top = p[self.y0[r0:r1]]
        bot = p[self.y1[r0:r1]]
        bot -= top
        bot *= self.fy[r0:r1]
        top += bot
        return top

    def rows(self, r0: int, r1: int):
        res = [self._one(p, r0, r1) for p in self.planes]
        return res[0] if self.single else res

    def full(self):
        outs = [np.empty((self.H, self.W), dtype=np.float32) for _ in self.planes]

        def k(r0: int, r1: int) -> None:
            for o, p in zip(outs, self.planes):
                o[r0:r1] = self._one(p, r0, r1)

        run_blocks(k, self.H, self.W)
        return outs[0] if self.single else outs


# ---------------------------------------------------------------------------
# Resampling (sparse separable matrices: cost ∝ kernel taps, any size)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=128)
def resample_matrix(n_out: int, n_in: int) -> sparse.csr_matrix:
    """Separable triangle-filter resampling matrix (n_out × n_in), Pillow BILINEAR semantics."""
    scale = n_in / n_out
    fs = max(scale, 1.0)
    c = (np.arange(n_out) + 0.5) * scale
    lo = np.floor(c - fs).astype(np.int64)
    width = int(np.ceil(2 * fs)) + 2
    j = lo[:, None] + np.arange(width)[None, :]
    w = np.maximum(0.0, 1.0 - np.abs((j + 0.5 - c[:, None]) / fs))
    if scale > 1.0:  # downsampling: drop taps outside the image (renormalized below)
        w = np.where((j >= 0) & (j < n_in), w, 0.0)
    s = w.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    w = w / s
    rows = np.repeat(np.arange(n_out), width)
    m = sparse.csr_matrix(
        (w.ravel().astype(np.float32), (rows, np.clip(j, 0, n_in - 1).ravel())), shape=(n_out, n_in)
    )
    m.sum_duplicates()
    m.eliminate_zeros()
    return m


def resize(a: np.ndarray, h: int, w: int) -> np.ndarray:
    """Triangle-filter resize of an H×W or H×W×C float32 array (see :func:`resample_matrix`)."""
    a = np.asarray(a, dtype=np.float32)
    H, W = a.shape[:2]
    if (H, W) == (h, w):
        return a.copy()
    extra = a.shape[2:]
    c = int(np.prod(extra)) if extra else 1
    t = a.reshape(H, W * c)
    if H != h:
        t = np.asarray(resample_matrix(h, H) @ t, dtype=np.float32)
    if W != w:
        t3 = t.reshape(h, W, c).transpose(1, 0, 2).reshape(W, h * c)
        t3 = np.asarray(resample_matrix(w, W) @ t3, dtype=np.float32)
        t = t3.reshape(w, h, c).transpose(1, 0, 2)
    return np.ascontiguousarray(t.reshape((h, w) + extra), dtype=np.float32)


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
def _sep_filter(a: np.ndarray, fn: Callable[..., None]) -> np.ndarray:
    a = np.ascontiguousarray(a, dtype=np.float32)
    tmp = np.empty_like(a)
    out = np.empty_like(a)
    H, W = a.shape
    run_rows(lambda s, e: fn(a[s:e], 1, tmp[s:e]), H)
    run_rows(lambda s, e: fn(tmp[:, s:e], 0, out[:, s:e]), W)
    return out


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 2-D float32 array (large sigmas via a reduced grid)."""
    if sigma < 0.25:
        return np.array(a, dtype=np.float32, copy=True)
    H, W = a.shape
    if sigma > 6.0 and min(H, W) > 32:
        f = max(2, int(sigma / 3.0))
        low = block_mean(a, f)
        low = ndimage.gaussian_filter(low, np.sqrt(max(0.1, (sigma / f) ** 2 - 0.25)), mode="reflect", truncate=3.0)
        return LowMap(low, H, W, f).full()
    if a.size < 150_000:
        return ndimage.gaussian_filter(np.asarray(a, dtype=np.float32), sigma, mode="reflect", truncate=3.0)
    return _sep_filter(
        a, lambda src, ax, out: ndimage.gaussian_filter1d(src, sigma, axis=ax, output=out, mode="reflect", truncate=3.0)
    )


def box_blur(a: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * int(radius) + 1
    if size <= 1:
        return np.array(a, dtype=np.float32, copy=True)
    a = np.asarray(a, dtype=np.float32)
    if a.size < 400_000:
        return ndimage.uniform_filter(a, size, mode="reflect")
    return _sep_filter(a, lambda src, ax, out: ndimage.uniform_filter1d(src, size, axis=ax, output=out, mode="reflect"))


def guided_coeffs(guide: np.ndarray, srcs: Sequence[np.ndarray | None], radius: int,
                  eps: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Guided-filter coefficients (box-averaged a, b) on a (reduced) grid (He et al. 2010).

    ``q = a·I + b`` with the full-resolution guide I gives the fast guided filter
    (He & Sun 2015). ``None`` in ``srcs`` means self-guided.
    """
    g = np.asarray(guide, dtype=np.float32)
    r = max(1, int(radius))
    m_i = box_blur(g, r)
    v_i = box_blur(g * g, r)
    v_i -= m_i * m_i
    np.maximum(v_i, F32(0.0), out=v_i)
    den = v_i + F32(eps)
    out = []
    for src in srcs:
        if src is None:
            m_p, cov = m_i, v_i
        else:
            pl = np.asarray(src, dtype=np.float32)
            m_p = box_blur(pl, r)
            cov = box_blur(g * pl, r) - m_i * m_p
        a = cov / den
        b = m_p - a * m_i
        out.append((box_blur(a, r), box_blur(b, r)))
    return out


def guided_filter(guide: np.ndarray, srcs: Sequence[np.ndarray | None], radius: float, eps: float,
                  subsample: float = 1.0) -> list[np.ndarray]:
    """Fast guided filter: coefficients on a grid reduced by ``subsample``, applied at full resolution."""
    guide = np.asarray(guide, dtype=np.float32)
    H, W = guide.shape
    f = max(1, int(round(subsample)))
    gl = block_mean(guide, f)
    sl = [None if s is None else block_mean(s, f) for s in srcs]
    r = max(1, int(round(radius / f)))
    res = []
    for a, b in guided_coeffs(gl, sl, r, eps):
        if f > 1:
            A, B = LowMap([a, b], H, W, f).full()
            q = A * guide + B
        else:
            q = a * guide + b
        res.append(q.astype(np.float32))
    return res
