"""White balance (Kelvin/tint ↔ camera multipliers, auto WB) and blown-highlight handling.

Kelvin model
------------
The illuminant chromaticity for a correlated colour temperature T is taken from
the Planckian locus (Kim et al. 2002 cubic-spline approximation, 1667–25000 K;
the same polynomials in 1/T extend smoothly up to 50000 K). ``tint`` moves the
illuminant perpendicular to the locus in CIE 1960 (u, v): ``Duv = tint / 3000``
(±150 → ±0.05; positive = greener illuminant, i.e. the image turns magenta, the
usual raw-converter convention). xy → XYZ (Y = 1) → camera via:

* ``cam_xyz`` (COLR flags bit1; DNG ColorMatrix, XYZ → camera) when valid;
* otherwise an approximation: ``cam = diag(1/daylight_wb) · rgb_cam⁻¹ · M(XYZ→sRGB)``,
  i.e. it assumes ``daylight_wb`` neutralizes D65 and ``rgb_cam`` maps
  white-balanced camera RGB to linear sRGB; without a usable ``rgb_cam`` the
  camera is modelled as wide-gamut ProPhoto primaries normalized to D65 (so
  low Kelvin values stay representable).

Multipliers neutralize that illuminant (``mul = 1 / cam``), normalized G = 1.
:func:`multipliers_to_kelvin` inverts the forward model with a damped Newton
solve in (mired, tint), so the round trip is exact up to solver tolerance.

Highlights
----------
``clip``: legacy hard clip to 1 after WB (blown areas neutral white).
``blend``: gradual fade to white as a pixel approaches sensor saturation.
``reconstruct``: saturated channels are rebuilt from the unsaturated ones using
the chromaticity of nearby unclipped pixels (push-pull hole filling on a
reduced grid), keeping the values above 1 so negative exposure / highlights
recover the detail.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .common import F32, block_mean, run_blocks, smoothstep

__all__ = [
    "kelvin_to_xy", "kelvin_to_multipliers", "multipliers_to_kelvin", "cam_from_xyz_matrix",
    "estimate_auto_wb", "handle_highlights", "KELVIN_RANGE", "TINT_RANGE",
]

KELVIN_RANGE = (2000.0, 50000.0)
TINT_RANGE = (-150.0, 150.0)
TINT_DUV = 1.0 / 3000.0

_PP_XYZ = np.array([[0.7976749, 0.1351917, 0.0313534], [0.2880402, 0.7118741, 0.0000857], [0.0, 0.0, 0.8252100]])
XYZ_TO_SRGB = np.array(
    [[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560], [0.0556434, -0.2040259, 1.0572252]]
)


def _planck_xy(t: float) -> tuple[float, float]:
    t = float(np.clip(t, 1000.0, 1e6))
    it = 1.0 / t
    if t <= 4000.0:
        x = -0.2661239e9 * it**3 - 0.2343589e6 * it**2 + 0.8776956e3 * it + 0.179910
    else:
        x = -3.0258469e9 * it**3 + 2.1070379e6 * it**2 + 0.2226347e3 * it + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def _xy_to_uv(x: float, y: float) -> tuple[float, float]:
    d = -2.0 * x + 12.0 * y + 3.0
    return 4.0 * x / d, 6.0 * y / d


def _uv_to_xy(u: float, v: float) -> tuple[float, float]:
    d = 2.0 * u - 8.0 * v + 4.0
    return 3.0 * u / d, 2.0 * v / d


def kelvin_to_xy(kelvin: float, tint: float = 0.0) -> tuple[float, float]:
    """Illuminant chromaticity for (T, tint) — see module docstring."""
    t = float(kelvin)
    u, v = _xy_to_uv(*_planck_xy(t))
    if tint:
        dt = max(1.0, t * 1e-3)
        u1, v1 = _xy_to_uv(*_planck_xy(t + dt))
        u0, v0 = _xy_to_uv(*_planck_xy(t - dt))
        du, dv = u1 - u0, v1 - v0
        n = np.hypot(du, dv) or 1.0
        nu, nv = -dv / n, du / n
        if nv < 0:  # normal pointing to the green side (+v)
            nu, nv = -nu, -nv
        u += float(tint) * TINT_DUV * nu
        v += float(tint) * TINT_DUV * nv
    return _uv_to_xy(u, v)


def _valid3(v: Sequence[float]) -> np.ndarray | None:
    try:
        a = np.array([float(x) for x in list(v)[:3]], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return a if a.shape == (3,) and np.all(np.isfinite(a)) and np.all(a > 0) else None


def cam_from_xyz_matrix(color: Any) -> tuple[np.ndarray, str]:
    """(3×3 XYZ → camera-raw matrix, source) — ``"cam_xyz"`` or ``"approx"``."""
    flags = int(getattr(color, "flags", 0) or 0)
    if flags & 2:
        m = np.array(getattr(color, "cam_xyz", ()), dtype=np.float64)
        if m.size == 9 and np.all(np.isfinite(m)):
            m = m.reshape(3, 3)
            if abs(np.linalg.det(m)) > 1e-9:
                return m, "cam_xyz"
    dl = _valid3(getattr(color, "daylight_wb", (1, 1, 1))) if color is not None else None
    if dl is None:
        dl = np.ones(3)
    dl = dl / dl[1]
    if flags & 1:
        m = np.array(getattr(color, "rgb_cam", ()), dtype=np.float64)
        if m.size == 9 and np.all(np.isfinite(m)) and abs(np.linalg.det(m.reshape(3, 3))) > 1e-9                 and not np.allclose(m.reshape(3, 3), np.eye(3)):
            return np.diag(1.0 / dl) @ np.linalg.inv(m.reshape(3, 3)) @ XYZ_TO_SRGB, "approx"
    # no colour data: a wide-gamut (ProPhoto-like) camera, neutral under D65 after daylight_wb
    pp = np.linalg.inv(_PP_XYZ)
    d65 = pp @ _xyz(0.31271, 0.32902)
    return np.diag(1.0 / dl) @ np.diag(1.0 / d65) @ pp, "approx"


def _xyz(x: float, y: float) -> np.ndarray:
    y = max(y, 1e-6)
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


def kelvin_to_multipliers(kelvin: float, tint: float, color: Any) -> tuple[float, float, float]:
    """Camera WB multipliers (R, G, B; G = 1) neutralizing a (T, tint) illuminant."""
    m, _ = cam_from_xyz_matrix(color)
    cam = m @ _xyz(*kelvin_to_xy(kelvin, tint))
    cam = np.maximum(cam, 1e-6)
    mul = cam[1] / cam
    return float(mul[0]), 1.0, float(mul[2])


def _logratios(kelvin: float, tint: float, m: np.ndarray) -> np.ndarray:
    cam = np.maximum(m @ _xyz(*kelvin_to_xy(kelvin, tint)), 1e-9)
    return np.log(np.array([cam[1] / cam[0], cam[1] / cam[2]]))


def multipliers_to_kelvin(mults: Any, color: Any) -> tuple[float, float]:
    """Inverse of :func:`kelvin_to_multipliers` → (kelvin, tint), clamped to the DEV ranges.

    Accepts the arguments in either order (``(mults, color)`` or ``(color, mults)``).
    """
    if hasattr(mults, "as_shot_wb") and not hasattr(color, "as_shot_wb"):
        mults, color = color, mults
    mv = _valid3(mults)
    if mv is None:
        return 5500.0, 0.0
    m, _ = cam_from_xyz_matrix(color)
    target = np.log(np.array([mv[0] / mv[1], mv[2] / mv[1]]))
    # coarse search on the locus, then Newton in (mired, tint)
    best, bt = None, 5500.0
    for t in np.geomspace(1500.0, 60000.0, 90):
        d = float(np.sum((_logratios(t, 0.0, m) - target) ** 2))
        if best is None or d < best:
            best, bt = d, float(t)
    mired, tint = 1e6 / bt, 0.0
    for _ in range(40):
        t = 1e6 / mired
        f = _logratios(t, tint, m) - target
        if float(np.max(np.abs(f))) < 1e-10:
            break
        hm, ht = max(0.05, mired * 1e-4), 0.05
        jm = (_logratios(1e6 / (mired + hm), tint, m) - _logratios(1e6 / (mired - hm), tint, m)) / (2 * hm)
        jt = (_logratios(t, tint + ht, m) - _logratios(t, tint - ht, m)) / (2 * ht)
        jac = np.stack([jm, jt], axis=1)
        try:
            step = np.linalg.solve(jac, f)
        except np.linalg.LinAlgError:
            break
        step = np.clip(step, [-60.0, -60.0], [60.0, 60.0])
        mired = float(np.clip(mired - step[0], 1e6 / 80000.0, 1e6 / 1000.0))
        tint = float(np.clip(tint - step[1], -400.0, 400.0))
    k = float(np.clip(1e6 / mired, *KELVIN_RANGE))
    return k, float(np.clip(tint, *TINT_RANGE))


# ---------------------------------------------------------------------------
# Auto white balance
# ---------------------------------------------------------------------------
def estimate_auto_wb(cam: np.ndarray, sat_level: float = 0.98) -> tuple[float, float, float] | None:
    """Robust grey-world estimate on pre-WB camera RGB (N×3 or H×W×3, values in [0, 1]).

    Excludes (nearly) saturated and the darkest 5 % of pixels, combines the grey-
    world mean (p = 1: higher Minkowski norms are biased by the different noise of
    the averaged green sites) with a white-patch estimate of the brightest valid
    pixels (geometric blend 0.8 / 0.2). Returns (R, 1, B).
    """
    a = np.asarray(cam, dtype=np.float64).reshape(-1, 3)
    mx = a.max(axis=1)
    ok = (mx < sat_level) & (a.min(axis=1) > 0.0)
    if ok.sum() < 16:
        ok = mx < sat_level
    v = a[ok]
    if v.shape[0] < 4:
        return None
    lum = v.mean(axis=1)
    lo = np.quantile(lum, 0.05)
    v = v[lum > max(lo, 1e-4)]
    if v.shape[0] < 4:
        return None
    sog = np.mean(v, axis=0)
    lum = v.mean(axis=1)
    top = v[lum >= np.quantile(lum, 0.98)]
    wp = top.mean(axis=0) if top.shape[0] else sog
    est = np.exp(0.8 * np.log(np.maximum(sog, 1e-9)) + 0.2 * np.log(np.maximum(wp, 1e-9)))
    if not np.all(np.isfinite(est)) or np.any(est <= 0):
        return None
    mul = est[1] / est
    mul = np.clip(mul, 1e-3, 64.0)
    return float(mul[0]), 1.0, float(mul[2])


# ---------------------------------------------------------------------------
# Highlights
# ---------------------------------------------------------------------------
def _push_pull(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Fills holes (den == 0) of num/den with values propagated from coarser levels."""
    levels = [(num, den)]
    n, d = num, den
    while min(n.shape[:2]) > 1:
        h, w = n.shape[:2]
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        pad = ((0, h2 * 2 - h), (0, w2 * 2 - w)) + ((0, 0),) * (n.ndim - 2)
        n = np.pad(n, pad, mode="edge").reshape(h2, 2, w2, 2, *n.shape[2:]).sum(axis=(1, 3))
        d = np.pad(d, pad[:2], mode="edge").reshape(h2, 2, w2, 2).sum(axis=(1, 3))
        levels.append((n, d))
    n, d = levels[-1]
    est = n / np.maximum(d, 1e-12)[..., None]
    for n, d in reversed(levels[:-1]):
        h, w = d.shape
        up = np.repeat(np.repeat(est, 2, axis=0), 2, axis=1)[:h, :w]
        wgt = np.clip(d, 0.0, 1.0)[..., None]
        est = np.where(d[..., None] > 0, n / np.maximum(d, 1e-12)[..., None], up) * wgt + up * (1 - wgt)
    return est


def handle_highlights(x: np.ndarray, sat: np.ndarray, mode: str) -> np.ndarray:
    """Applies ``blend`` / ``reconstruct`` in place on white-balanced camera RGB.

    ``x``: H×W×3 float32 (WB applied, *not* clipped). ``sat``: per-channel
    saturation level in the same units (3,). ``clip`` is handled by the caller.
    """
    H, W, _ = x.shape
    sat = np.asarray(sat, dtype=np.float32)
    if mode == "blend":
        inv = (1.0 / sat).astype(np.float32)

        def run(r0: int, r1: int) -> None:
            v = x[r0:r1]
            m = np.maximum(np.maximum(v[..., 0] * inv[0], v[..., 1] * inv[1]), v[..., 2] * inv[2])
            np.minimum(v, F32(1.0), out=v)
            w = smoothstep(0.8, 1.0, m)
            top = np.maximum(np.maximum(v[..., 0], v[..., 1]), v[..., 2])
            for c in range(3):
                ch = v[..., c]
                ch += w * (top - ch)

        run_blocks(run, H, W)
        return x
    if mode != "reconstruct":
        return x
    thr = (0.985 * sat).astype(np.float32)
    anyc = (x[..., 0] >= thr[0]) | (x[..., 1] >= thr[1]) | (x[..., 2] >= thr[2])
    if not anyc.any():
        return x
    # mean colour of valid (unclipped) pixels on a reduced grid, holes filled
    f = max(1, int(round(max(H, W) / 256)))
    vf = (~anyc).astype(np.float32)
    num = np.stack([block_mean(x[..., c] * vf, f) for c in range(3)], axis=-1).astype(np.float64)
    den = block_mean(vf, f).astype(np.float64)
    chrom = _push_pull(num, den)
    chrom = (chrom / np.maximum(chrom.sum(axis=2, keepdims=True), 1e-12)).astype(np.float32)
    ys, xs = np.nonzero(anyc)
    qq = _bilinear_at(chrom, ys, xs, f)
    v = x[ys, xs]
    cl = v >= thr
    unc = ~cl
    has = unc.any(axis=1)
    sum_u = np.where(unc, v, 0.0).sum(axis=1)
    sum_q = np.where(unc, qq, 0.0).sum(axis=1)
    total = sum_u / np.maximum(sum_q, 1e-6)
    est = total[:, None] * qq
    # smooth transition near the threshold, never darker than the clipped value
    wgt = np.clip((v - 0.9 * sat) / (0.085 * sat), 0.0, 1.0)
    rec = np.maximum(v, v + wgt * (est - v))
    rec = np.where(has[:, None] & cl, rec, v)
    allc = ~has  # every channel clipped: neutral at the highest level reached
    if allc.any():
        rec[allc] = np.max(v[allc], axis=1, keepdims=True)
    x[ys, xs] = rec.astype(np.float32)
    return x


def _bilinear_at(low: np.ndarray, ys: np.ndarray, xs: np.ndarray, f: int) -> np.ndarray:
    h, w = low.shape[:2]
    u = (ys + 0.5) / f - 0.5
    t = (xs + 0.5) / f - 0.5
    y0 = np.clip(np.floor(u), 0, h - 1).astype(np.intp)
    x0 = np.clip(np.floor(t), 0, w - 1).astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy = np.clip(u - y0, 0, 1)[:, None].astype(np.float32)
    fx = np.clip(t - x0, 0, 1)[:, None].astype(np.float32)
    top = low[y0, x0] + (low[y0, x1] - low[y0, x0]) * fx
    bot = low[y1, x0] + (low[y1, x1] - low[y1, x0]) * fx
    return top + (bot - top) * fy
