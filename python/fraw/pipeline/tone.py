"""Tone: transfer curves, contrast, parametric + point curves, local tone mapping, dehaze.

Local tone stage (linear light, after exposure)
-----------------------------------------------
* ``I = sqrt(Y)`` (Rec.709 luminance, perceptual-ish domain).
* Base layer ``B = guided(I)`` — self-guided fast guided filter (radius ≈ 2 % of
  the long side, eps 0.008) computed on a reduced grid and applied with the
  full-resolution guide: edge-aware, so the gain field follows strong edges and
  does not halo, while texture inside regions is left in the detail layer.
* Highlights / shadows: gain ``2**(shadows·1.5·(1 − ss(0, .5, Yb)) +
  highlights·1.0·ss(.25, 1, Yb))`` with ``Yb = B²`` — the DEV v1 curves, but
  driven by the local base luminance instead of the pixel (local tone mapping).
* Clarity: midtone-weighted boost of the large-scale detail ``I − B``.
  Texture: band-pass ``G(σ₁)(I) − G(σ₂)(I)`` (σ₁ ≈ 0.7 px, σ₂ ≈ 4 px at full
  resolution) — medium frequencies, not noise, not large structures.
* Whites / blacks: global curves on the (already gained) luminance in
  ``p = Y**(1/2.2)``: ``p + 0.2·whites·p⁴`` and ``p + 0.12·blacks·(1 − p)⁴``
  (positive blacks lift as a neutral offset, negative ones crush multiplicatively).
* Calibration shadow tint: green gain ``2**(−0.35·tint·(1 − ss(0, .3, Y)))``.

All of it is one per-pixel gain (plus an optional neutral offset) applied to
RGB, so hue and saturation ratios are preserved.

Dehaze: simplified dark-channel prior (He et al. 2009) on a ~256 px grid:
dark channel → atmospheric light A (brightest 0.1 %) → transmission
``t = 1 − ω·dark(I/A)`` (ω = 0.85·amount) refined with a guided filter, upsampled bilinearly;
``J = (I − A)/max(t, 0.1) + A``. Negative amounts blend towards A (adds haze).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from .common import F32, REC709, LowMap, block_mean, gaussian_blur, guided_coeffs, run_blocks, smoothstep

__all__ = [
    "srgb_encode", "transfer", "contrast_curve", "monotone_spline", "parametric_curve",
    "tone_tables", "local_tone", "dehaze", "dehaze_airlight", "SHADOWS_EV", "HIGHLIGHTS_EV",
]

SHADOWS_EV = 1.5
HIGHLIGHTS_EV = 1.0
CONTRAST_GAMMA = 2.2
MID_GREY = 0.18
GAIN_LUT = 4096
WB_LUT_MAX = 2.0
WB_LUT = 8192


def srgb_encode(x: np.ndarray) -> np.ndarray:
    """Official sRGB OETF (piecewise), for x in [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    lo = x * np.float32(12.92)
    hi = np.float32(1.055) * np.power(np.maximum(x, np.float32(0.0031308)), np.float32(1 / 2.4)) - np.float32(0.055)
    return np.where(x <= 0.0031308, lo, hi).astype(np.float32)


def transfer(x: np.ndarray, mode: str = "srgb", gamma: float = 2.2) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    if mode == "srgb":
        return srgb_encode(x)
    if mode == "power":
        return np.power(x, np.float32(1.0 / gamma)).astype(np.float32)
    if mode == "linear":
        return x
    raise ValueError(f"unknown gamma_mode {mode!r}")


def contrast_curve(x: np.ndarray, contrast: float) -> np.ndarray:
    """DEV v1 S-curve in p = x**(1/2.2) around middle grey. x is clipped to [0, 1]."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    if not contrast:
        return x
    a = np.float32(2.0**contrast)
    inv = np.float32(1.0 / CONTRAST_GAMMA)
    p = np.power(x, inv)
    p0 = np.float32(MID_GREY**inv)
    lo = p0 * np.power(np.minimum(p, p0) / p0, a)
    hi = 1.0 - (1.0 - p0) * np.power(np.maximum(1.0 - p, 0.0) / (1.0 - p0), a)
    q = np.where(p < p0, lo, hi)
    return np.power(np.clip(q, 0.0, 1.0), np.float32(CONTRAST_GAMMA)).astype(np.float32)


# ---------------------------------------------------------------------------
# Curves (output-encoded domain)
# ---------------------------------------------------------------------------
def monotone_spline(points: Sequence[Sequence[float]], x: np.ndarray) -> np.ndarray:
    """Piecewise-cubic Hermite with Fritsch–Carlson slopes (no overshoot), clamped to [0, 1].

    Outside the first/last control point the curve is held constant (the usual
    tone-curve behaviour).
    """
    pts = np.asarray(points, dtype=np.float64)
    xk, yk = pts[:, 0], pts[:, 1]
    n = len(xk)
    x = np.asarray(x, dtype=np.float64)
    if n == 2:
        t = np.clip((x - xk[0]) / max(xk[1] - xk[0], 1e-9), 0, 1)
        return np.clip(yk[0] + t * (yk[1] - yk[0]), 0, 1)
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.empty(n)
    m[0], m[-1] = d[0], d[-1]
    m[1:-1] = (d[:-1] + d[1:]) * 0.5
    for k in range(n - 1):
        if d[k] == 0.0:
            m[k] = m[k + 1] = 0.0
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
    for k in range(n - 1):
        if d[k] == 0:
            continue
        a, b = m[k] / d[k], m[k + 1] / d[k]
        s = a * a + b * b
        if s > 9.0:
            tau = 3.0 / np.sqrt(s)
            m[k], m[k + 1] = tau * a * d[k], tau * b * d[k]
    xc = np.clip(x, xk[0], xk[-1])
    i = np.clip(np.searchsorted(xk, xc, side="right") - 1, 0, n - 2)
    hh = h[i]
    t = (xc - xk[i]) / hh
    t2, t3 = t * t, t * t * t
    y = ((2 * t3 - 3 * t2 + 1) * yk[i] + (t3 - 2 * t2 + t) * hh * m[i] + (-2 * t3 + 3 * t2) * yk[i + 1]
         + (t3 - t2) * hh * m[i + 1])
    return np.clip(y, 0.0, 1.0)


_PARAM_CENTERS = (0.125, 0.375, 0.625, 0.875)  # shadows, darks, lights, highlights


def parametric_curve(x: np.ndarray, shadows: float, darks: float, lights: float, highlights: float) -> np.ndarray:
    """4-region parametric curve: smooth bumps centred at 1/8, 3/8, 5/8, 7/8, endpoints fixed.

    Each slider moves its region by up to ±0.15 (output units); the curve is made
    monotone (running maximum) and clamped to [0, 1].
    """
    x = np.asarray(x, dtype=np.float64)
    y = x.copy()
    for amt, c in zip((shadows, darks, lights, highlights), _PARAM_CENTERS):
        if amt:
            bump = np.exp(-0.5 * ((x - c) / 0.11) ** 2) * (4.0 * x * (1.0 - x)) ** 0.5
            y = y + 0.15 * amt * bump
    y = np.maximum.accumulate(np.clip(y, 0.0, 1.0))
    return y


def tone_tables(p: Any, n: int = 4096) -> np.ndarray | None:
    """(3, n) per-channel table on the output-encoded domain for parametric + point curves.

    ``None`` when every curve is the identity.
    """
    from ..devparams import is_identity_curve

    par = (p.curve_p_shadows, p.curve_p_darks, p.curve_p_lights, p.curve_p_highlights)
    curves = (p.curve_master, p.curve_red, p.curve_green, p.curve_blue)
    if not any(par) and all(is_identity_curve(c) for c in curves):
        return None
    x = np.linspace(0.0, 1.0, n)
    y = parametric_curve(x, *par) if any(par) else x
    if not is_identity_curve(p.curve_master):
        y = monotone_spline(p.curve_master, y)
    out = np.empty((3, n), dtype=np.float32)
    for c, cur in enumerate(curves[1:]):
        out[c] = monotone_spline(cur, y) if not is_identity_curve(cur) else y
    return out


# ---------------------------------------------------------------------------
# Local tone
# ---------------------------------------------------------------------------
def _hs_table(highlights: float, shadows: float) -> np.ndarray:
    y = np.linspace(0.0, 1.0, GAIN_LUT, dtype=np.float32)
    ev = np.zeros_like(y)
    if shadows:
        ev += F32(shadows * SHADOWS_EV) * (1.0 - smoothstep(0.0, 0.5, y))
    if highlights:
        ev += F32(highlights * HIGHLIGHTS_EV) * smoothstep(0.25, 1.0, y)
    return np.exp2(ev).astype(np.float32)


def _wb_tables(whites: float, blacks: float) -> tuple[np.ndarray, np.ndarray]:
    """(gain, offset) tables on Y ∈ [0, WB_LUT_MAX] for whites/blacks."""
    y = np.linspace(0.0, WB_LUT_MAX, WB_LUT)
    p = np.power(y, 1 / 2.2)
    q = p + 0.2 * whites * np.power(np.minimum(p, 1.0), 4) if whites else p
    q = np.maximum(q, 0.0)
    y1 = np.power(q, 2.2)
    gain = np.where(y > 1e-7, y1 / np.maximum(y, 1e-7), 1.0)
    off = np.zeros_like(y)
    if blacks:
        p1 = np.power(y1, 1 / 2.2)
        q2 = p1 + 0.12 * blacks * np.power(np.clip(1.0 - p1, 0.0, 1.0), 4)
        y2 = np.power(np.maximum(q2, 0.0), 2.2)
        if blacks > 0:
            off = y2 - y1
        else:
            gain = gain * np.where(y1 > 1e-7, y2 / np.maximum(y1, 1e-7), 0.0)
    return gain.astype(np.float32), off.astype(np.float32)


def _sqrt_lum(lin: np.ndarray) -> np.ndarray:
    """I = sqrt(clip(Y, 0, 4)) as an H×W float32 map (row-block kernel)."""
    H, W, _ = lin.shape
    I = np.empty((H, W), dtype=np.float32)

    def k(r0: int, r1: int) -> None:
        v = lin[r0:r1]
        y = v[..., 0] * REC709[0]
        y += v[..., 1] * REC709[1]
        y += v[..., 2] * REC709[2]
        np.maximum(y, F32(0.0), out=y)
        np.minimum(y, F32(4.0), out=y)
        np.sqrt(y, out=I[r0:r1])

    run_blocks(k, H, W)
    return I


def local_tone(lin: np.ndarray, p: Any, scale: float, long_side: float, draft: bool = False) -> None:
    """Highlights/shadows/whites/blacks/clarity/texture/shadow tint — in place on linear RGB.

    ``scale``: output pixels per full-resolution pixel; ``long_side``: long side of
    the final frame in *output* pixels (the base-layer radius is relative to it).
    """
    hl, sh, wh, bl = float(p.highlights), float(p.shadows), float(p.whites), float(p.blacks)
    cl, tx, st = float(p.clarity), float(p.texture), float(p.cal_shadows_tint)
    if not any((hl, sh, wh, bl, cl, tx, st)):
        return
    H, W, _ = lin.shape
    I = _sqrt_lum(lin)
    base_map = None
    if hl or sh or cl:
        radius = max(1.0, 0.02 * long_side)
        f = max(1, int(round(radius / 4.0)))
        il = block_mean(I, f)
        (a, b), = guided_coeffs(il, [None], max(1, int(round(radius / f))), 0.008)
        base_map = LowMap([a, b], H, W, f)
    tex_map = tex_full = None
    if tx:
        s2 = max(1.0, 4.0 * scale)
        f2 = max(1, int(s2 / 1.5))
        low = block_mean(I, f2)
        low = ndimage.gaussian_filter(low, np.sqrt(max(0.1, (s2 / f2) ** 2 - 0.08)), mode="reflect", truncate=3.0)
        tex_map = LowMap(low, H, W, f2)
        s1 = 0.7 * scale
        if s1 >= 0.5 and not draft:
            tex_full = gaussian_blur(I, s1)
    hs_tab = _hs_table(hl, sh) if (hl or sh) else None
    wb_g, wb_o = _wb_tables(wh, bl) if (wh or bl) else (None, None)
    gm = F32(GAIN_LUT - 1)
    wm = F32((WB_LUT - 1) / WB_LUT_MAX)
    wtop = F32(WB_LUT - 1)
    kcl = F32(cl * 1.2)
    ktx = F32(tx * 1.6)
    kst = F32(-0.35 * st)

    def k(r0: int, r1: int) -> None:
        v = lin[r0:r1]
        i = I[r0:r1]
        g = None
        b = None
        if base_map is not None:
            A, B = base_map.rows(r0, r1)
            b = A * i
            b += B
            np.maximum(b, F32(0.0), out=b)
        if hs_tab is not None:
            yb = b * b
            yb *= gm
            np.minimum(yb, gm, out=yb)
            g = hs_tab[yb.astype(np.int32)]
        if cl or tx:
            d = None
            if cl:
                mid = F32(4.0) * b * (F32(1.0) - b)
                np.maximum(mid, F32(0.0), out=mid)
                np.minimum(mid, F32(1.0), out=mid)
                d = kcl * mid * (i - b)
            if tx:
                band = (tex_full[r0:r1] if tex_full is not None else i) - tex_map.rows(r0, r1)
                band *= ktx
                d = band if d is None else d + band
            r = d / np.maximum(i, F32(1e-3))
            r += F32(1.0)
            np.maximum(r, F32(0.5), out=r)
            np.minimum(r, F32(2.0), out=r)
            r *= r
            g = r if g is None else g * r
        off = None
        if wb_g is not None:
            y1 = i * i
            if g is not None:
                y1 *= g
            y1 *= wm
            np.minimum(y1, wtop, out=y1)
            idx = y1.astype(np.int32)
            gw = wb_g[idx]
            g = gw if g is None else g * gw
            if bl > 0:
                off = wb_o[idx]
        if g is not None:
            v *= g[..., None]
        if off is not None:
            v += off[..., None]
        if st:
            y = i * i
            v[..., 1] *= np.exp2(kst * (F32(1.0) - smoothstep(0.0, 0.3, y)))

    run_blocks(k, H, W)


# ---------------------------------------------------------------------------
# Dehaze
# ---------------------------------------------------------------------------
def _small(lin: np.ndarray, side: int = 256) -> tuple[np.ndarray, int]:
    H, W, _ = lin.shape
    f = max(1, int(round(max(H, W) / side)))
    return np.maximum(block_mean(lin, f), 0.0), f


def dehaze_airlight(lin: np.ndarray) -> np.ndarray:
    """Atmospheric light A (3,) from the brightest 0.1 % of the dark channel."""
    sm, _ = _small(lin)
    h, w, _ = sm.shape
    dark = ndimage.minimum_filter(sm.min(axis=2), size=max(3, int(round(max(h, w) * 0.03)) | 1))
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    a = sm.reshape(-1, 3)[idx].mean(axis=0)
    return np.maximum(a, 1e-3).astype(np.float32)


def dehaze(lin: np.ndarray, amount: float, airlight: np.ndarray | None = None) -> None:
    """In-place dehaze of linear RGB (see module docstring)."""
    if not amount:
        return
    H, W, _ = lin.shape
    A = dehaze_airlight(lin) if airlight is None else np.asarray(airlight, dtype=np.float32)
    if amount < 0:
        hz = F32(min(1.0, -amount) * 0.5)
        Ah = (A * hz).astype(np.float32)

        def k_haze(r0: int, r1: int) -> None:
            v = lin[r0:r1]
            v *= F32(1.0) - hz
            v += Ah

        run_blocks(k_haze, H, W)
        return
    sm, f = _small(lin)
    h, w, _ = sm.shape
    norm = sm / A[None, None, :]
    size = max(3, int(round(max(h, w) * 0.03)) | 1)
    dark = ndimage.minimum_filter(norm.min(axis=2), size=size)
    omega = F32(0.85 * min(1.0, amount))
    t = (F32(1.0) - omega * dark).astype(np.float32)
    guide = np.sqrt(np.clip(sm @ REC709, 0.0, 4.0)).astype(np.float32)
    (a, b), = guided_coeffs(guide, [t], max(2, int(size * 1.5)), 1e-3)
    t = np.clip(a * guide + b, 0.1, 1.0).astype(np.float32)
    inv = LowMap(F32(1.0) / t, H, W, f)

    def k_dh(r0: int, r1: int) -> None:
        v = lin[r0:r1]
        v -= A
        v *= inv.rows(r0, r1)[..., None]
        v += A
        np.maximum(v, F32(0.0), out=v)

    run_blocks(k_dh, H, W)
