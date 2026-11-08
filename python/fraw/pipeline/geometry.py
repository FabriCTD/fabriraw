"""Lens optics + geometry: one inverse mapping, one resampling pass.

Coordinate frames (all continuous, in full-resolution pixels, pixel centres at
``i + 0.5``):

* **base**: the EXIF-oriented image (W₀ × H₀).
* **canvas**: base after ``rotate_90`` (clockwise) and ``flip_h`` / ``flip_v``
  (exact, no interpolation). Size W_c × H_c.
* **final**: the crop rectangle of the transformed canvas (``crop_*`` are
  normalized to the canvas, i.e. to the image *after* rotation/perspective).

For every output pixel the inverse chain is evaluated: final → canvas →
inverse(scale · straighten rotation · perspective homography) about the canvas
centre → lens distortion ``r_src = r · (1 + k r²)`` (``k = −0.15 · lens_distortion``,
r normalized to the half diagonal; + corrects barrel, − pincushion) → lateral
CA (R and B sampled at radius × (1 + 0.003 · ca)) → canvas → base → source
array. Lens vignetting is a radial gain ``2**(lens_vignette · r²)`` (+1 → +1 EV
in the corners). Radial lens operations commute with the exact 90° rotations
and flips, so they are evaluated in canvas orientation.

Straighten/perspective/distortion automatically zoom in just enough that no
empty (black) border remains ("constrain crop"); ``perspective_scale`` then
scales on top of that (values < 1 can reveal black borders).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .common import F32, run_rows

__all__ = ["GeoPlan", "make_plan", "warp", "canvas_from_final", "ASPECTS"]

ASPECTS = {"1:1": 1.0, "4:5": 0.8, "5:4": 1.25, "3:2": 1.5, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16,
           "4:3": 4 / 3, "3:4": 0.75}
PERSPECTIVE_K = 0.35
DISTORTION_K = 0.15
CA_K = 0.003
MIN_CROP = 0.01


@dataclass
class GeoPlan:
    base_w: int
    base_h: int
    k90: int = 0
    flip_h: bool = False
    flip_v: bool = False
    canvas_w: int = 0
    canvas_h: int = 0
    minv: np.ndarray | None = None  # canvas-centred → image-centred (homogeneous 3×3)
    k_dist: float = 0.0
    ca_r: float = 0.0
    ca_b: float = 0.0
    lens_vig: float = 0.0
    crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # canvas units
    key: tuple = field(default=())

    @property
    def warp(self) -> bool:
        return self.minv is not None or bool(self.k_dist or self.ca_r or self.ca_b or self.lens_vig)

    @property
    def cropped(self) -> bool:
        x0, y0, x1, y1 = self.crop
        return (x0, y0, x1, y1) != (0.0, 0.0, float(self.canvas_w), float(self.canvas_h))

    @property
    def identity(self) -> bool:
        return not self.warp and not self.cropped and self.k90 == 0 and not self.flip_h and not self.flip_v

    @property
    def final_w(self) -> float:
        return self.crop[2] - self.crop[0]

    @property
    def final_h(self) -> float:
        return self.crop[3] - self.crop[1]

    def final_size(self) -> tuple[int, int]:
        return max(1, int(round(self.final_w))), max(1, int(round(self.final_h)))

    # ------------------------------------------------------------ mapping
    def canvas_to_base(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact inverse of rot90/flip for points (continuous coords)."""
        w, h = float(self.canvas_w), float(self.canvas_h)
        if self.flip_h:
            x = w - x
        if self.flip_v:
            y = h - y
        for _ in range(self.k90):  # undo one clockwise quarter turn: dims (w, h) → (h, w)
            x, y = y, w - x
            w, h = h, w
        return x, y

    def base_to_canvas(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, h = float(self.base_w), float(self.base_h)
        for _ in range(self.k90):  # clockwise: (x, y) → (h − y, x)
            x, y = h - y, x
            w, h = h, w
        if self.flip_h:
            x = w - x
        if self.flip_v:
            y = h - y
        return x, y

    def optical(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Canvas coords → centred, undistorted-to-distorted source coords (before CA)."""
        cx, cy = self.canvas_w * 0.5, self.canvas_h * 0.5
        qx, qy = X - cx, Y - cy
        if self.minv is not None:
            m = self.minv
            w = m[2, 0] * qx + m[2, 1] * qy + m[2, 2]
            w = np.where(np.abs(w) < 1e-9, 1e-9, w)
            px = (m[0, 0] * qx + m[0, 1] * qy + m[0, 2]) / w
            py = (m[1, 0] * qx + m[1, 1] * qy + m[1, 2]) / w
        else:
            px, py = qx, qy
        if self.k_dist:
            rn2 = 1.0 / (0.25 * (self.canvas_w**2 + self.canvas_h**2))
            f = 1.0 + self.k_dist * (px * px + py * py) * rn2
            px, py = px * f, py * f
        return px, py

    def final_to_base(self, X: np.ndarray, Y: np.ndarray, channel: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Final-frame coords → base coords (for channel ``channel``'s CA scaling)."""
        px, py = self.optical(X + self.crop[0], Y + self.crop[1])
        k = 1.0 + (self.ca_r if channel == 0 else self.ca_b if channel == 2 else 0.0) * CA_K
        x = px * k + self.canvas_w * 0.5
        y = py * k + self.canvas_h * 0.5
        return self.canvas_to_base(x, y)


def _crop_rect(p: Any, cw: int, ch: int) -> tuple[float, float, float, float]:
    l, t, r, b = (float(p.crop_left), float(p.crop_top), float(p.crop_right), float(p.crop_bottom))
    if r - l < MIN_CROP:
        r = min(1.0, l + MIN_CROP)
        l = r - MIN_CROP
    if b - t < MIN_CROP:
        b = min(1.0, t + MIN_CROP)
        t = b - MIN_CROP
    x0, x1, y0, y1 = l * cw, r * cw, t * ch, b * ch
    asp = str(getattr(p, "crop_aspect", "free"))
    ratio = cw / ch if asp == "original" else ASPECTS.get(asp)
    if ratio:
        w, h = x1 - x0, y1 - y0
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if w / h > ratio:
            w = h * ratio
        else:
            h = w / ratio
        x0, x1, y0, y1 = mx - w / 2, mx + w / 2, my - h / 2, my + h / 2
    return x0, y0, x1, y1


def _inside(plan: GeoPlan, z: float, fwd: np.ndarray, bx: np.ndarray, by: np.ndarray) -> bool:
    s = np.diag([1.0 / z, 1.0 / z, 1.0])
    plan.minv = np.linalg.inv(fwd) @ s
    px, py = plan.optical(bx, by)
    hw, hh = plan.canvas_w * 0.5, plan.canvas_h * 0.5
    kk = 1.0 + max(abs(plan.ca_r), abs(plan.ca_b)) * CA_K
    return bool(np.all(np.abs(px) * kk <= hw + 1e-6) and np.all(np.abs(py) * kk <= hh + 1e-6))


def make_plan(p: Any, base_w: int, base_h: int, apply_crop: bool = True) -> GeoPlan:
    """Geometry plan for DEV params ``p`` on a base (oriented) image of base_w × base_h."""
    k90 = (int(str(p.rotate_90)) // 90) % 4
    cw, ch = (base_h, base_w) if k90 % 2 else (base_w, base_h)
    plan = GeoPlan(base_w, base_h, k90, bool(p.flip_h), bool(p.flip_v), cw, ch)
    plan.k_dist = -DISTORTION_K * float(p.lens_distortion)
    plan.ca_r, plan.ca_b = float(p.ca_red_cyan), float(p.ca_blue_yellow)
    plan.lens_vig = float(p.lens_vignette)
    theta = math.radians(float(p.rotate_deg))
    pv, ph = float(p.perspective_vertical), float(p.perspective_horizontal)
    us = float(p.perspective_scale)
    fill = bool(theta or pv or ph or plan.k_dist or plan.ca_r or plan.ca_b)
    if fill or abs(us - 1.0) > 1e-9:
        c, s = math.cos(theta), math.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        hp = np.eye(3)
        hp[2, 0] = -ph * PERSPECTIVE_K / (cw * 0.5)
        hp[2, 1] = pv * PERSPECTIVE_K / (ch * 0.5)
        fwd = rot @ hp
        z = 1.0
        if fill:
            n = 33
            t = np.linspace(0.0, 1.0, n)
            bx = np.concatenate([t * cw, t * cw, np.zeros(n), np.full(n, float(cw))])
            by = np.concatenate([np.zeros(n), np.full(n, float(ch)), t * ch, t * ch])
            lo, hi = 0.25, 8.0
            if _inside(plan, lo, fwd, bx, by):
                z = lo
            else:
                for _ in range(48):
                    mid = math.sqrt(lo * hi)
                    if _inside(plan, mid, fwd, bx, by):
                        hi = mid
                    else:
                        lo = mid
                z = hi
        zz = z * us
        plan.minv = np.linalg.inv(fwd) @ np.diag([1.0 / zz, 1.0 / zz, 1.0])
    if apply_crop:
        plan.crop = _crop_rect(p, cw, ch)
    else:
        plan.crop = (0.0, 0.0, float(cw), float(ch))
    plan.key = (
        base_w, base_h, k90, plan.flip_h, plan.flip_v, None if plan.minv is None else tuple(np.round(plan.minv, 12).ravel()),
        plan.k_dist, plan.ca_r, plan.ca_b, plan.lens_vig, tuple(round(v, 6) for v in plan.crop),
    )
    return plan


def canvas_from_final(plan: GeoPlan, xn: float, yn: float) -> tuple[float, float]:
    return plan.crop[0] + xn * plan.final_w, plan.crop[1] + yn * plan.final_h


def warp(
    src: np.ndarray,
    plan: GeoPlan,
    x0: float,
    y0: float,
    out_w: int,
    out_h: int,
    s_out: float,
    src_scale: float,
    src_off: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Resamples ``src`` (base orientation) into the final frame (bilinear).

    Output pixel (i, j) covers final coords ``(x0 + (j + .5)/s_out, y0 + (i + .5)/s_out)``.
    ``src`` covers base coords starting at ``src_off`` with ``src_scale`` source
    pixels per full-resolution pixel. Outside samples are black.
    """
    sh, sw = src.shape[:2]
    flat = src.reshape(-1, 3)
    out = np.empty((out_h, out_w, 3), dtype=np.float32)
    xs = (x0 + (np.arange(out_w) + 0.5) / s_out).astype(np.float64)
    ys = (y0 + (np.arange(out_h) + 0.5) / s_out).astype(np.float64)
    ca = bool(plan.ca_r or plan.ca_b)
    rn2 = 1.0 / (0.25 * (plan.canvas_w**2 + plan.canvas_h**2))

    def coords(X: np.ndarray, Y: np.ndarray, ch: int) -> tuple[np.ndarray, np.ndarray]:
        bx, by = plan.final_to_base(X, Y, ch)
        return (bx - src_off[0]) * src_scale - 0.5, (by - src_off[1]) * src_scale - 0.5

    def sample(sx: np.ndarray, sy: np.ndarray, chans: slice | int) -> np.ndarray:
        inside = (sx >= -0.5) & (sx <= sw - 0.5) & (sy >= -0.5) & (sy <= sh - 0.5)
        sx = np.clip(sx, 0.0, sw - 1.0)
        sy = np.clip(sy, 0.0, sh - 1.0)
        ix = np.minimum(sx.astype(np.int64), sw - 2) if sw > 1 else np.zeros(sx.shape, np.int64)
        iy = np.minimum(sy.astype(np.int64), sh - 2) if sh > 1 else np.zeros(sy.shape, np.int64)
        fx = (sx - ix).astype(np.float32)
        fy = (sy - iy).astype(np.float32)
        if sw == 1:
            fx[:] = 0
        if sh == 1:
            fy[:] = 0
        i00 = iy * sw + ix
        dx = 1 if sw > 1 else 0
        dy = sw if sh > 1 else 0
        if isinstance(chans, int):
            col = flat[:, chans]
            a, b = col[i00], col[i00 + dx]
            c, d = col[i00 + dy], col[i00 + dy + dx]
            top = a + (b - a) * fx
            bot = c + (d - c) * fx
            v = top + (bot - top) * fy
            return np.where(inside, v, F32(0.0))
        fx, fy = fx[:, None], fy[:, None]
        a, b = flat[i00], flat[i00 + dx]
        c, d = flat[i00 + dy], flat[i00 + dy + dx]
        top = a + (b - a) * fx
        bot = c + (d - c) * fx
        v = top + (bot - top) * fy
        return np.where(inside[:, None], v, F32(0.0))

    def run(r0: int, r1: int) -> None:
        X, Y = np.meshgrid(xs, ys[r0:r1])
        X, Y = X.ravel(), Y.ravel()
        if ca:
            res = np.empty((X.size, 3), dtype=np.float32)
            for c in range(3):
                sx, sy = coords(X, Y, c)
                res[:, c] = sample(sx, sy, c)
        else:
            sx, sy = coords(X, Y, 1)
            res = sample(sx, sy, slice(None))
        if plan.lens_vig:
            px, py = plan.optical(X + plan.crop[0], Y + plan.crop[1])
            res *= np.exp2(plan.lens_vig * (px * px + py * py) * rn2).astype(np.float32)[:, None]
        out[r0:r1] = res.reshape(r1 - r0, out_w, 3)

    run_rows(run, out_h, parts=max(1, min(32, out_h // 8)))
    return out
