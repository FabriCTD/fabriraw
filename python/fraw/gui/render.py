"""Qt-free rendering helpers for the GUI (testable without a display).

* :func:`oriented_size`, :func:`stored_to_oriented`, :func:`oriented_rect_to_stored`
  — EXIF orientation geometry, consistent with :func:`fraw.processor.apply_orientation`.
* :func:`develop_region` — full-resolution develop of a sub-rectangle (canvas
  detail tile at ≥ 50 % zoom) using the public FRAWProcessor on a crop of the
  same FrawFile (shares the mosaic, no copy).
* :func:`neutral_wb_from_patch` — eyedropper: custom WB that neutralizes a patch.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..format import FrawFile
from ..processor import FRAWProcessor, to_uint8

__all__ = [
    "oriented_size",
    "stored_to_oriented",
    "oriented_to_stored",
    "oriented_rect_to_stored",
    "stored_rect_to_oriented",
    "develop_region",
    "neutral_wb_from_patch",
    "patch_cam_rgb",
    "base_crop",
]


def base_crop(f: FrawFile) -> tuple[int, int, int, int]:
    """(left, top, width, height) of the active area in sensor coordinates."""
    h = f.head
    if h.crop_left or h.crop_top or h.crop_width or h.crop_height:
        return h.crop_left, h.crop_top, h.crop_width, h.crop_height
    return 0, 0, h.width, h.height


def oriented_size(w: int, h: int, orientation: int) -> tuple[int, int]:
    return (h, w) if int(orientation) in (5, 6, 7, 8) else (w, h)


def oriented_to_stored(yo: int, xo: int, w: int, h: int, o: int) -> tuple[int, int]:
    """Oriented pixel (yo, xo) → stored pixel (ys, xs); (w, h) = stored size."""
    o = int(o)
    if o == 2:
        return yo, w - 1 - xo
    if o == 3:
        return h - 1 - yo, w - 1 - xo
    if o == 4:
        return h - 1 - yo, xo
    if o == 5:
        return xo, yo
    if o == 6:
        return h - 1 - xo, yo
    if o == 7:
        return h - 1 - xo, w - 1 - yo
    if o == 8:
        return xo, w - 1 - yo
    return yo, xo


def stored_to_oriented(ys: int, xs: int, w: int, h: int, o: int) -> tuple[int, int]:
    """Stored pixel (ys, xs) → oriented pixel (yo, xo); (w, h) = stored size."""
    o = int(o)
    if o == 2:
        return ys, w - 1 - xs
    if o == 3:
        return h - 1 - ys, w - 1 - xs
    if o == 4:
        return h - 1 - ys, xs
    if o == 5:
        return xs, ys
    if o == 6:
        return xs, h - 1 - ys
    if o == 7:
        return w - 1 - xs, h - 1 - ys
    if o == 8:
        return w - 1 - xs, ys
    return ys, xs


def _rect_map(fn, x0: int, y0: int, x1: int, y1: int, w: int, h: int, o: int) -> tuple[int, int, int, int]:
    """Maps the half-open rect [x0,x1)×[y0,y1) through ``fn`` (corner pixels)."""
    pts = [fn(y, x, w, h, o) for y in (y0, y1 - 1) for x in (x0, x1 - 1)]
    ys = [p[0] for p in pts]
    xs = [p[1] for p in pts]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def oriented_rect_to_stored(rect: tuple[int, int, int, int], w: int, h: int, o: int) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) in oriented space → same area in stored space; (w, h) = stored size."""
    return _rect_map(oriented_to_stored, *rect, w, h, o)


def stored_rect_to_oriented(rect: tuple[int, int, int, int], w: int, h: int, o: int) -> tuple[int, int, int, int]:
    return _rect_map(stored_to_oriented, *rect, w, h, o)


def develop_region(
    f: FrawFile, params, rect_oriented: tuple[float, float, float, float], margin: int = 8
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Full-resolution develop of an oriented-space rect (x0, y0, x1, y1).

    Returns (uint8 H×W×3, (x0, y0, x1, y1) actually rendered, oriented space).
    """
    bl, bt, bw, bh = base_crop(f)
    o = f.head.orientation
    ow, oh = oriented_size(bw, bh, o)
    x0 = max(0, int(np.floor(rect_oriented[0])) - margin)
    y0 = max(0, int(np.floor(rect_oriented[1])) - margin)
    x1 = min(ow, int(np.ceil(rect_oriented[2])) + margin)
    y1 = min(oh, int(np.ceil(rect_oriented[3])) + margin)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise ValueError("region too small")
    sx0, sy0, sx1, sy1 = oriented_rect_to_stored((x0, y0, x1, y1), bw, bh, o)
    head = dataclasses.replace(
        f.head, crop_left=bl + sx0, crop_top=bt + sy0, crop_width=sx1 - sx0, crop_height=sy1 - sy0
    )
    sub = dataclasses.replace(f, head=head)
    img = FRAWProcessor(sub).develop(params)
    out_rect = stored_rect_to_oriented((sx0, sy0, sx1, sy1), bw, bh, o)
    return np.ascontiguousarray(to_uint8(img)), out_rect


def patch_cam_rgb(proc: FRAWProcessor, xo: float, yo: float, size: int = 5) -> np.ndarray:
    """Mean pre-WB camera RGB of a ``size``×``size`` superpixel patch (2·size CFA px)
    centred on oriented image pixel (xo, yo)."""
    h, w = proc.norm.shape
    ys, xs = oriented_to_stored(int(yo), int(xo), w, h, proc.orientation)
    half = size  # CFA pixels on each side → size superpixels across
    y0 = max(0, min(h - 2 * size, ys - half)) & ~1
    x0 = max(0, min(w - 2 * size, xs - half)) & ~1
    patch = proc.norm[y0 : y0 + 2 * size, x0 : x0 + 2 * size]
    acc = np.zeros(3)
    cnt = np.zeros(3)
    for i, c in enumerate(proc.pattern):
        v = patch[i >> 1 :: 2, i & 1 :: 2]
        if v.size:
            acc[c] += float(v.mean())
            cnt[c] += 1
    return acc / np.maximum(cnt, 1)


def neutral_wb_from_patch(cam_rgb: np.ndarray) -> tuple[float, float, float] | None:
    """Mean camera RGB (pre-WB) of a patch → [R, G, B] multipliers that make it grey (G = 1)."""
    m = np.asarray(cam_rgb, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    if not np.all(np.isfinite(m)) or np.any(m <= 1e-6):
        return None
    if np.any(m >= 0.999):  # clipped channel: no reliable neutral
        return None
    return (float(m[1] / m[0]), 1.0, float(m[1] / m[2]))
