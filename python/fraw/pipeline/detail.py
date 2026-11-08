"""Noise reduction and sharpening (output-encoded domain, luminance/chroma split).

The DEV defaults are ``sharpen_amount`` = 0 and ``nr_color`` = 0: the baseline
rendering is the plain demosaic with no extra filtering (identical to DEV v1),
and the strength is the value itself (0 = no effect).

Colour NR: Y/Cb/Cr split (``Cb = B − Y``, ``Cr = R − Y``); each chroma plane
is filtered with a guided filter guided by Y (chroma edges that coincide with
luminance edges survive), radius 2…12 px (full-res), on a reduced grid (the eye's
chroma acuity is low), and the filtered chroma is upsampled bilinearly.
``nr_color_detail`` lowers the regularization (keeps more chroma detail).

Luminance NR: self-guided filter on Y (radius 1…3 px, eps ∝ strength²);
``nr_luminance_detail`` adds back part of the removed detail.

Sharpening: unsharp mask on Y with σ = radius (full-res px, ≥ 0.5 output px);
the high-pass ``d`` is soft-compressed ``d / (1 + |d|/τ)`` with
``τ = 0.01 + 0.2·detail²`` (low detail → strong edges only, fewer halos);
``sharpen_masking`` restricts it to edges via a gradient mask. Luminance changes
are added to R, G and B equally (no colour fringes).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import F32, REC709, LowMap, block_mean, gaussian_blur, guided_coeffs, run_blocks, smoothstep

__all__ = ["sharpen_strength", "color_nr_strength", "noise_reduction", "sharpen"]

SHARPEN_BASE = 0.0
NR_COLOR_BASE = 0.0


def sharpen_strength(p: Any) -> float:
    return float(p.sharpen_amount) - SHARPEN_BASE


def color_nr_strength(p: Any) -> float:
    return max(0.0, (float(p.nr_color) - NR_COLOR_BASE) / (1.0 - NR_COLOR_BASE))


def _lum_map(img: np.ndarray) -> np.ndarray:
    H, W, _ = img.shape
    y = np.empty((H, W), dtype=np.float32)

    def k(r0: int, r1: int) -> None:
        v = img[r0:r1]
        t = v[..., 0] * REC709[0]
        t += v[..., 1] * REC709[1]
        t += v[..., 2] * REC709[2]
        y[r0:r1] = t

    run_blocks(k, H, W)
    return y


def noise_reduction(img: np.ndarray, p: Any, scale: float) -> np.ndarray:
    """In-place chroma + luminance noise reduction; returns ``img``."""
    cs = color_nr_strength(p)
    ls = float(p.nr_luminance)
    if cs <= 0 and ls <= 0:
        return img
    H, W, _ = img.shape
    cmap = lmap = None
    if cs > 0:
        # chroma acuity is low: filter on a reduced grid (edge-aware there, guided by Y)
        # and upsample the filtered chroma itself
        radius = max(1.0, (2.0 + 10.0 * cs) * scale)
        f = max(2, int(round(radius / 2.0)))
        if scale < 1.0:  # preview: coarser grid (≈ export appearance at this size)
            f = max(f, 4)
        f = min(f, max(1, min(H, W) // 4))
        low = block_mean(img, f)
        yl = low @ REC709
        cbl = low[..., 2] - yl
        crl = low[..., 0] - yl
        eps = (0.0004 + 0.004 * (1.0 - float(p.nr_color_detail))) * (0.5 + cs)
        (a1, b1), (a2, b2) = guided_coeffs(yl, [cbl, crl], max(1, int(round(radius / f))), eps)
        cmap = LowMap([a1 * yl + b1, a2 * yl + b2], H, W, f)
        kc = F32(min(1.0, 1.25 * cs + 0.2))
    if ls > 0:
        radius = max(1.0, (1.0 + 2.0 * ls) * scale)
        f = 2 if min(H, W) >= 8 else 1
        yl = block_mean(_lum_map(img), f)
        eps = (0.006 + 0.05 * ls) ** 2
        (a, b), = guided_coeffs(yl.astype(np.float32), [None], max(1, int(round(radius / f))), eps)
        lmap = LowMap([a, b], H, W, f)
        keep = F32(0.6 * float(p.nr_luminance_detail) * (1.0 - 0.5 * ls))
    r709, b709 = F32(REC709[0]), F32(REC709[2])
    ig = F32(1.0 / REC709[1])

    def k(r0: int, r1: int) -> None:
        v = img[r0:r1]
        y = v[..., 0] * REC709[0]
        y += v[..., 1] * REC709[1]
        y += v[..., 2] * REC709[2]
        cb = v[..., 2] - y
        cr = v[..., 0] - y
        if cmap is not None:
            qcb, qcr = cmap.rows(r0, r1)
            q = qcb
            q -= cb
            q *= kc
            cb += q
            q = qcr
            q -= cr
            q *= kc
            cr += q
        if lmap is not None:
            c = lmap.rows(r0, r1)
            q = c[0] * y
            q += c[1]
            y -= q
            y *= keep
            y += q
        rr = y + cr
        bb = y + cb
        gg = (y - r709 * rr - b709 * bb) * ig
        v[..., 0] = rr
        v[..., 1] = gg
        v[..., 2] = bb

    run_blocks(k, H, W)
    return img


def sharpen(img: np.ndarray, p: Any, scale: float, amount: float | None = None,
            radius: float | None = None) -> None:
    """In-place unsharp mask on luminance (see module docstring)."""
    a = sharpen_strength(p) if amount is None else float(amount)
    if abs(a) < 1e-6:
        return
    r = float(p.sharpen_radius) if radius is None else float(radius)
    sig = r * scale
    att = min(1.0, sig / 0.5)  # sub-pixel radius at this scale: weaker, as when downsampling
    sig = max(0.5, sig)
    H, W, _ = img.shape
    y = _lum_map(img)
    blur = gaussian_blur(y, sig)
    k = F32(2.0 * a * att)
    tau = F32(0.01 + 0.2 * float(p.sharpen_detail) ** 2)
    masking = float(p.sharpen_masking) if a > 0 else 0.0
    t0 = F32(0.04 * masking)
    t1 = F32(2.0 * 0.04 * masking + 0.01)

    def kern(r0: int, r1: int) -> None:
        d = y[r0:r1] - blur[r0:r1]
        if a > 0:
            d /= F32(1.0) + np.abs(d) / tau
            if masking > 0:
                bl = blur[max(0, r0 - 1) : min(H, r1 + 1)]
                o = r0 - max(0, r0 - 1)
                gy = np.abs(np.diff(bl, axis=0, append=bl[-1:]))[o : o + (r1 - r0)]
                gx = np.abs(np.diff(blur[r0:r1], axis=1, append=blur[r0:r1, -1:]))
                d *= smoothstep(float(t0), float(t1), gx + gy)
        d *= k
        img[r0:r1] += d[..., None]

    run_blocks(kern, H, W)
