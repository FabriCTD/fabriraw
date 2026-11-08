"""FabriRAW development engine (SPEC §5 reference pipeline, DEV v2 "Motor Pro").

Pipeline order (identical in :meth:`FRAWProcessor.develop`, :meth:`~FRAWProcessor.preview`
and :meth:`~FRAWProcessor.render_region`; radii are expressed in full-resolution
pixels or relative to the final frame and scaled to the working resolution):

1. **Linear camera**: crop to the HEAD active area,
   ``x = (raw − black[cfa]) / (white − black[cfa])`` clipped to [0, 1].
2. **highlight_mode** + **WB** (as_shot / daylight / custom / kelvin, then
   ``temperature_shift`` / ``tint_shift``; multipliers normalized min = 1):
   ``clip`` multiplies and clips at 1 (legacy: blown areas neutral white);
   ``blend`` fades towards white near saturation; ``reconstruct`` rebuilds the
   saturated channels from the unsaturated ones with the chromaticity of
   neighbouring unclipped pixels and keeps values > 1 (see :mod:`fraw.pipeline.wb`).
3. **Demosaic** (``bilinear`` | ``ahd``), or a 2×2 superpixel for the preview / ``half_size``.
4. **Colour matrix**: ``saturation · calibration · 2**exposure · rgb_cam`` fused in
   one 3×3 matrix (all linear). Calibration rotates/scales the ProPhoto
   primaries (:mod:`fraw.pipeline.color`). *Saturation* keeps its DEV v1 place
   (a luminance-preserving linear map in scene light) so v1 looks are unchanged;
   it commutes with the luminance gains below.
5. **Optics + geometry** (one inverse mapping, one bilinear resampling;
   :mod:`fraw.pipeline.geometry`): lens vignetting, radial distortion, lateral
   CA (R/B radial scale), then rotate_90/flip (exact), straighten, perspective
   (+ scale, automatic fill without black borders) and the normalized crop on
   the transformed image. Because WB and the colour matrix are per-pixel linear
   maps they commute with the resampling; the engine therefore resamples the
   camera-space image *before* step 4 (and, in the preview, before WB) so the
   expensive warp is cached independently of colour/tone edits.
6. **Defringe** (purple/green fringes on high-contrast edges, linear).
7. **Dehaze** (simplified dark-channel prior, linear).
8. **Local tone** (linear): highlights/shadows driven by an edge-aware base layer
   (fast guided filter on √Y: local tone mapping without halos), clarity (large
   scale detail, midtones), texture (band-pass), whites/blacks (global end
   points), calibration shadow tint (:mod:`fraw.pipeline.tone`).
9. **Contrast** (DEV v1 S-curve around 0.18) → **output curve** (sRGB / power /
   linear) → **parametric curve** (4 regions) → **point curves** (Fritsch–Carlson
   monotone cubic; master then R/G/B) — all per channel, fused in one LUT. From
   here on the image is in the *output-encoded* domain, so the point curves act
   in the perceptual space of the chosen output curve.
10. **HSL** 8 bands + **vibrance** (:mod:`fraw.pipeline.hsl`).
11. **Colour grading** wheels with blending/balance (:mod:`fraw.pipeline.grading`).
12. **Noise reduction** (chroma: guided filter on Cb/Cr guided by Y; luminance:
    self-guided with detail) and **sharpening** (USM on Y, radius/detail/masking)
    (:mod:`fraw.pipeline.detail`).
13. **Post-crop vignette** and **grain** (final-frame coordinates; deterministic
    per-image seed) (:mod:`fraw.pipeline.effects`).
14. Clip to [0, 1] and quantize.

With every setting at its default the output is bit-identical to the DEV v1
engine (the defaults of ``sharpen_amount`` and ``nr_color`` are the baseline,
see :mod:`fraw.pipeline.detail`).

Preview: a cached half-size superpixel pyramid; with default-ish settings
(only WB/exposure/contrast/saturation/output curve) the v1 fused-LUT fast
path runs (≈ 12 ms at 1600 px). Otherwise the stage pipeline runs with a
per-stage cache keyed by the parameters of that stage and all earlier ones, so
changing a late setting (grain, vignette, grading…) only recomputes the tail.
``draft=True`` renders at ≤ 0.25 Mpx (integer factor ≥ 2 below the preview;
no NR/sharpening/defringe, texture without its fine scale) and upsamples
bilinearly — for slider drags.

EXIF orientation (HEAD.orientation) is applied before the DEV geometry.
"""

from __future__ import annotations

import math
import os
import struct
import threading
import zlib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from . import demosaic as _dm
from .devparams import SCHEMA, DevParams, is_identity_curve
from .format import FrawFile
from .pipeline import color as _color
from .pipeline import detail as _detail
from .pipeline import effects as _fx
from .pipeline import geometry as _geo
from .pipeline import grading as _grading
from .pipeline import hsl as _hsl
from .pipeline import tone as _tone
from .pipeline import wb as _wb
from .pipeline.autoadjust import auto_tone as _auto_tone
from .pipeline.common import luminance, quantize8, run_chunks
from .pipeline.icc import build_exif, srgb_icc
from .pipeline.tone import contrast_curve as _contrast_curve
from .pipeline.tone import srgb_encode, transfer
from .pipeline.wb import kelvin_to_multipliers, multipliers_to_kelvin

__all__ = [
    "FRAWProcessor",
    "srgb_encode",
    "transfer",
    "apply_orientation",
    "to_uint8",
    "to_uint16",
    "write_png16",
    "write_tiff16",
    "histogram",
    "clipping_mask",
    "kelvin_to_multipliers",
    "multipliers_to_kelvin",
    "REC709",
    "TEMPERATURE_K",
    "TINT_K",
]

REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
TEMPERATURE_K = 0.5  # R·exp(+k·t), B·exp(−k·t): t = ±1 → ×1.65 / ×0.61
TINT_K = 0.5  # G·exp(−k·tint): +1 → magenta, −1 → green (Lightroom convention)
SHADOWS_EV = _tone.SHADOWS_EV
HIGHLIGHTS_EV = _tone.HIGHLIGHTS_EV
MID_GREY = 0.18
GAIN_LUT_SIZE = 4096  # highlights/shadows gain vs. luminance (smooth → small LUT)
LUT_SIZE = 65536  # linear-domain LUT; 4096 was ±6 levels off in the shadows of power curves
_CHUNK = 32768  # preview pixels per work item (fits in L2 cache)
# amount of the export USM (≈ 0.6 px); screen/glossy/matte are accepted as aliases
DRAFT_PIXELS = 250_000  # working pixels of a draft preview
OUTPUT_SHARPEN = {"none": 0.0, "low": 0.25, "standard": 0.5, "high": 0.85,
                  "screen": 0.5, "glossy": 0.6, "matte": 0.85}

# --------------------------------------------------------------------- stage keys
_WB_KEYS = ("wb_mode", "wb_custom", "wb_kelvin", "wb_kelvin_tint", "temperature_shift", "tint_shift")
_GEO_KEYS = ("rotate_90", "flip_h", "flip_v", "rotate_deg", "perspective_vertical", "perspective_horizontal",
             "perspective_scale", "crop_aspect", "crop_left", "crop_top", "crop_right", "crop_bottom",
             "lens_distortion", "lens_vignette", "ca_red_cyan", "ca_blue_yellow")
_B_KEYS = ("highlight_mode", "exposure_ev", "saturation", "cal_shadows_tint", "cal_red_hue", "cal_red_sat",
           "cal_green_hue", "cal_green_sat", "cal_blue_hue", "cal_blue_sat", "defringe", "dehaze",
           "highlights", "shadows", "whites", "blacks", "clarity", "texture")
_C_KEYS = ("contrast", "gamma_mode", "gamma", "curve_p_highlights", "curve_p_lights", "curve_p_darks",
           "curve_p_shadows", "curve_master", "curve_red", "curve_green", "curve_blue", "vibrance") + tuple(
    s.key for s in SCHEMA if s.group in ("hsl", "grading"))
_D_KEYS = ("sharpen_amount", "sharpen_radius", "sharpen_detail", "sharpen_masking", "nr_luminance",
           "nr_luminance_detail", "nr_color", "nr_color_detail")
_E_KEYS = ("vignette_amount", "vignette_midpoint", "vignette_roundness", "vignette_feather", "vignette_highlights",
           "grain_amount", "grain_size", "grain_roughness")
_PROC_KEYS = ("demosaic",)
STAGE_KEYS = {"wb": _WB_KEYS, "geometry": _GEO_KEYS, "scene": _B_KEYS, "color": _C_KEYS, "detail": _D_KEYS,
              "effects": _E_KEYS, "process": _PROC_KEYS}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
def _hl_sh_gain(y: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    """DEV v1 global highlights/shadows gain (kept for the v1 fast path)."""
    ev = np.float32(0)
    if shadows:
        ev = ev + np.float32(shadows * SHADOWS_EV) * (1.0 - _smoothstep(0.0, 0.5, y))
    if highlights:
        ev = ev + np.float32(highlights * HIGHLIGHTS_EV) * _smoothstep(0.25, 1.0, y)
    return np.exp2(ev).astype(np.float32)


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _saturation_matrix(s: float) -> np.ndarray:
    return _color.saturation_matrix(s)


def apply_orientation(img: np.ndarray, orientation: int) -> np.ndarray:
    """Transforms the stored image so that it displays upright (EXIF 1–8)."""
    o = int(orientation)
    if o == 2:
        r = img[:, ::-1]
    elif o == 3:
        r = img[::-1, ::-1]
    elif o == 4:
        r = img[::-1, :]
    elif o == 5:
        r = np.swapaxes(img, 0, 1)
    elif o == 6:
        r = np.rot90(img, -1)  # 90° clockwise
    elif o == 7:
        r = np.swapaxes(img[::-1, ::-1], 0, 1)
    elif o == 8:
        r = np.rot90(img, 1)  # 90° counter-clockwise
    else:
        return img
    return np.ascontiguousarray(r)


def _display_to_stored(o: int, x: np.ndarray, y: np.ndarray, wd: float, hd: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`apply_orientation` for points (continuous coords; display size wd × hd)."""
    if o == 2:
        return wd - x, y
    if o == 3:
        return wd - x, hd - y
    if o == 4:
        return x, hd - y
    if o == 5:
        return y, x
    if o == 6:
        return y, wd - x
    if o == 7:
        return hd - y, wd - x
    if o == 8:
        return hd - y, x
    return x, y


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png16(path: str | os.PathLike, img16: np.ndarray, icc: bytes | None = None) -> None:
    """16-bit/channel RGB PNG (color type 2, bit depth 16, filter 0), via zlib; optional iCCP."""
    a = np.ascontiguousarray(img16, dtype=">u2")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected H×W×3 uint16")
    h, w, _ = a.shape
    rows = np.empty((h, 1 + w * 6), dtype=np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = a.view(np.uint8).reshape(h, w * 6)
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
    if icc:
        data += _png_chunk(b"iCCP", b"sRGB\0\0" + zlib.compress(icc, 9))
    data += _png_chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + _png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def write_tiff16(path: str | os.PathLike, img16: np.ndarray, icc: bytes | None = None) -> None:
    """Baseline little-endian, uncompressed, 16-bit/channel RGB TIFF (single strip); optional ICC (tag 34675)."""
    a = np.ascontiguousarray(img16, dtype="<u2")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected H×W×3 uint16")
    h, w, _ = a.shape
    pix = a.tobytes()
    SHORT, LONG, RATIONAL, UNDEFINED = 3, 4, 5, 7
    bps = struct.pack("<3H", 16, 16, 16)
    res = struct.pack("<II", 300, 1)
    entries = [
        (256, LONG, 1, struct.pack("<I", w)),
        (257, LONG, 1, struct.pack("<I", h)),
        (258, SHORT, 3, bps),
        (259, SHORT, 1, struct.pack("<H", 1)),
        (262, SHORT, 1, struct.pack("<H", 2)),
        (273, LONG, 1, None),  # StripOffsets, patched below
        (277, SHORT, 1, struct.pack("<H", 3)),
        (278, LONG, 1, struct.pack("<I", h)),
        (279, LONG, 1, struct.pack("<I", len(pix))),
        (282, RATIONAL, 1, res),
        (283, RATIONAL, 1, res),
        (284, SHORT, 1, struct.pack("<H", 1)),
        (296, SHORT, 1, struct.pack("<H", 2)),
    ]
    if icc:
        entries.append((34675, UNDEFINED, len(icc), bytes(icc)))
    n = len(entries)
    ifd_off = 8
    ifd_size = 2 + n * 12 + 4
    extra_off = ifd_off + ifd_size
    extra = b""
    body = []
    offsets: dict[int, int] = {}
    for tag, typ, cnt, val in entries:
        if val is not None and len(val) > 4:
            offsets[tag] = extra_off + len(extra)
            extra += val
            if len(extra) & 1:
                extra += b"\0"
    pix_off = extra_off + len(extra)
    for tag, typ, cnt, val in entries:
        if tag == 273:
            val = struct.pack("<I", pix_off)
        if len(val) > 4:
            fld = struct.pack("<I", offsets[tag])
        else:
            fld = val.ljust(4, b"\0")
        body.append(struct.pack("<HHI", tag, typ, cnt) + fld)
    data = b"II*\0" + struct.pack("<I", ifd_off) + struct.pack("<H", n) + b"".join(body) + b"\0\0\0\0" + extra + pix
    with open(path, "wb") as f:
        f.write(data)


def histogram(img: np.ndarray) -> dict[str, Any]:
    """256-bin R/G/B/luminance histograms of a uint8 H×W×3 image + clipping fractions.

    ``clip_shadows`` / ``clip_highlights``: fraction of pixels with any channel at 0 / 255.
    """
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = to_uint8(a)
    flat = a.reshape(-1, 3)
    out: dict[str, Any] = {}
    for i, k in enumerate("rgb"):
        out[k] = np.bincount(flat[:, i], minlength=256)[:256].astype(np.int64)
    lum = (flat.astype(np.uint32) @ np.array([54, 183, 19], dtype=np.uint32) + 128) >> 8
    out["l"] = np.bincount(np.minimum(lum, 255).astype(np.int64), minlength=256)[:256].astype(np.int64)
    n = max(1, flat.shape[0])
    out["clip_shadows"] = float(np.count_nonzero((flat == 0).any(axis=1))) / n
    out["clip_highlights"] = float(np.count_nonzero((flat == 255).any(axis=1))) / n
    return out


def clipping_mask(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(shadow_mask, highlight_mask): any channel at 0 / at 255 (bool H×W)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = to_uint8(a)
    return (a == 0).any(axis=2), (a == 255).any(axis=2)


def _p_key(p: DevParams, keys: tuple[str, ...]) -> tuple:
    out = []
    for k in keys:
        v = getattr(p, k)
        if isinstance(v, (list, tuple)):
            v = tuple(tuple(x) if isinstance(x, (list, tuple)) else x for x in v)
        out.append(v)
    return tuple(out)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
class FRAWProcessor:
    """Develops a :class:`~fraw.format.FrawFile` (see module docstring)."""

    def __init__(self, frawfile: FrawFile):
        self.frawfile = frawfile
        head = frawfile.head
        self.head = head
        self.color = frawfile.color
        raw = np.asarray(frawfile.raw)
        if head.crop_left or head.crop_top or head.crop_width or head.crop_height:
            l, t, cw, ch = head.crop_left, head.crop_top, head.crop_width, head.crop_height
        else:
            l, t, cw, ch = 0, 0, head.width, head.height
        raw = raw[t : t + ch, l : l + cw]
        # CFA pattern / black levels re-indexed to the crop origin.
        src = [head.cfa_index_at(y + t, x + l) for y in (0, 1) for x in (0, 1)]
        self.pattern: tuple[int, int, int, int] = tuple(int(head.cfa_pattern[i]) for i in src)  # type: ignore[assignment]
        black = np.array([head.black_level[i] for i in src], dtype=np.float32).reshape(2, 2)
        bmap = np.tile(black, ((ch + 1) // 2, (cw + 1) // 2))[:ch, :cw]
        white = np.float32(head.white_level)
        norm = (raw.astype(np.float32) - bmap) / (white - bmap)
        np.clip(norm, 0.0, 1.0, out=norm)
        self.norm: np.ndarray = norm  # float32 (crop_h, crop_w), CFA, [0,1]
        self.orientation = int(head.orientation)
        self._preview_cache: dict[int, dict[str, Any]] = {}
        self._preview_lock = threading.RLock()  # preview buffers / caches are shared between calls
        self._sp: np.ndarray | None = None  # un-oriented superpixel (pre-WB)
        self._levels: dict[int, np.ndarray] = {}
        self._stages: OrderedDict[tuple, dict[str, tuple[Any, Any]]] = OrderedDict()
        self._plans: OrderedDict[tuple, _geo.GeoPlan] = OrderedDict()
        sample = np.ascontiguousarray(norm[:: max(1, norm.shape[0] // 61), :: max(1, norm.shape[1] // 67)])
        self.seed = zlib.crc32(sample.tobytes(), zlib.crc32(struct.pack("<II", *norm.shape))) & 0xFFFFFFFF

    @classmethod
    def from_path(cls, path: str | os.PathLike) -> "FRAWProcessor":
        return cls(FrawFile.read(path))

    # ------------------------------------------------------------- params
    def default_params(self) -> DevParams:
        return DevParams.from_dict(self.frawfile.dev)

    def _params(self, params: DevParams | Mapping[str, Any] | None) -> DevParams:
        if params is None:
            return self.default_params()
        return DevParams.coerce(params)

    @staticmethod
    def _valid_mul(v: Any) -> float | None:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) and f > 0 else None

    def wb_multipliers(self, params: DevParams | Mapping[str, Any] | None = None) -> np.ndarray:
        """Per-colour multipliers [R, G, B, G2] after mode + shifts, min == 1."""
        p = self._params(params)
        if p.wb_mode == "custom":
            base = [p.wb_custom[0], p.wb_custom[1], p.wb_custom[2], p.wb_custom[1]]
        elif p.wb_mode == "kelvin":
            r, g, b = kelvin_to_multipliers(p.wb_kelvin, p.wb_kelvin_tint, self.color)
            base = [r, g, b, g]
        else:
            src = self.color.as_shot_wb if p.wb_mode == "as_shot" else self.color.daylight_wb
            base = list(src)
        m = [self._valid_mul(v) for v in base[:4]]
        if m[0] is None or m[1] is None or m[2] is None:
            m = [1.0, 1.0, 1.0, 1.0]
        if m[3] is None:
            m[3] = m[1]
        mul = np.array(m, dtype=np.float64)
        mul[0] *= np.exp(TEMPERATURE_K * p.temperature_shift)
        mul[2] *= np.exp(-TEMPERATURE_K * p.temperature_shift)
        mul[1] *= np.exp(-TINT_K * p.tint_shift)
        mul[3] *= np.exp(-TINT_K * p.tint_shift)
        mul /= mul.min()
        return mul

    def _wb3(self, p: DevParams) -> np.ndarray:
        mul = self.wb_multipliers(p)
        return np.array([mul[0], 0.5 * (mul[1] + mul[3]), mul[2]], dtype=np.float64)

    def _site_multipliers(self, mul: np.ndarray) -> np.ndarray:
        """2×2 multipliers per CFA site: greens in a blue row get G2 (dcraw convention)."""
        site = np.empty(4, dtype=np.float32)
        for i, c in enumerate(self.pattern):
            if c == 1:
                site[i] = mul[3] if self.pattern[i ^ 1] == 2 else mul[1]
            else:
                site[i] = mul[0] if c == 0 else mul[2]
        return site.reshape(2, 2)

    def color_matrix(self) -> np.ndarray:
        if self.color.flags & 1:
            m = np.array(self.color.rgb_cam, dtype=np.float64).reshape(3, 3)
            if np.all(np.isfinite(m)):
                return m
        return np.eye(3)

    def _linear_matrix(self, p: DevParams) -> np.ndarray:
        """saturation · calibration · 2**ev · rgb_cam (all linear; see module docstring)."""
        ev = float(np.clip(p.exposure_ev, -4.0, 4.0))
        cal = _color.calibration_matrix(p.cal_red_hue, p.cal_red_sat, p.cal_green_hue, p.cal_green_sat,
                                        p.cal_blue_hue, p.cal_blue_sat)
        return _saturation_matrix(p.saturation) @ (cal @ (2.0**ev * self.color_matrix()))

    # ---------------------------------------------------------- features
    @staticmethod
    def _features(p: DevParams) -> set[str]:
        f: set[str] = set()
        if p.highlight_mode != "clip":
            f.add("highlight")
        if any((p.highlights, p.shadows, p.whites, p.blacks, p.clarity, p.texture, p.cal_shadows_tint)):
            f.add("localtone")
        if p.dehaze:
            f.add("dehaze")
        if p.defringe:
            f.add("defringe")
        if any((p.curve_p_highlights, p.curve_p_lights, p.curve_p_darks, p.curve_p_shadows)) or not all(
                is_identity_curve(getattr(p, k)) for k in ("curve_master", "curve_red", "curve_green", "curve_blue")):
            f.add("curves")
        if _hsl.hsl_tables(p) is not None or p.vibrance:
            f.add("hsl")
        if _grading.grading_table(p) is not None:
            f.add("grading")
        if _detail.color_nr_strength(p) > 0 or p.nr_luminance > 0:
            f.add("nr")
        if abs(_detail.sharpen_strength(p)) > 1e-6:
            f.add("sharpen")
        if p.vignette_amount:
            f.add("vignette")
        if p.grain_amount > 0:
            f.add("grain")
        return f

    # ------------------------------------------------------------ stages
    def _wb_cfa(self, p: DevParams, clip: bool = True) -> np.ndarray:
        site = self._site_multipliers(self.wb_multipliers(p))
        h, w = self.norm.shape
        smap = np.tile(site, ((h + 1) // 2, (w + 1) // 2))[:h, :w]
        out = self.norm * smap
        if clip:
            np.minimum(out, 1.0, out=out)  # highlight clip after WB (no magenta)
        return out

    def _superpixel(self, cfa: np.ndarray) -> np.ndarray:
        h2, w2 = cfa.shape[0] // 2, cfa.shape[1] // 2
        out = np.zeros((h2, w2, 3), dtype=np.float32)
        count = [0, 0, 0]
        for i, c in enumerate(self.pattern):
            out[..., c] += cfa[i >> 1 : 2 * h2 : 2, i & 1 : 2 * w2 : 2]
            count[c] += 1
        for c in range(3):
            if count[c] != 1:
                out[..., c] *= np.float32(1.0 / count[c])
        return out

    def _level(self, f: int) -> np.ndarray:
        """Pre-WB superpixel box-downscaled by ``f`` and EXIF-oriented (cached)."""
        lv = self._levels.get(f)
        if lv is not None:
            return lv
        if self._sp is None:
            self._sp = self._superpixel(self.norm)
        sp = self._sp
        h, w, _ = sp.shape
        if f > 1:
            hh, ww = max(1, h // f), max(1, w // f)
            sp = sp[: hh * f, : ww * f].reshape(hh, f, ww, f, 3).mean(axis=(1, 3), dtype=np.float32)
        lv = np.ascontiguousarray(apply_orientation(sp, self.orientation), dtype=np.float32)
        self._levels[f] = lv
        return lv

    def oriented_size(self) -> tuple[int, int]:
        """(w, h) of the EXIF-oriented full-resolution image (before DEV geometry)."""
        h, w = self.norm.shape
        return (h, w) if self.orientation in (5, 6, 7, 8) else (w, h)

    def _plan(self, p: DevParams, apply_crop: bool, base: tuple[int, int] | None = None) -> _geo.GeoPlan:
        bw, bh = base or self.oriented_size()
        key = (bw, bh, apply_crop) + _p_key(p, _GEO_KEYS)
        pl = self._plans.get(key)
        if pl is None:
            pl = _geo.make_plan(p, bw, bh, apply_crop)
            self._plans[key] = pl
            while len(self._plans) > 16:
                self._plans.popitem(last=False)
        return pl

    def output_size(self, params: DevParams | Mapping[str, Any] | None = None, apply_crop: bool = True) -> tuple[int, int]:
        """(w, h) of the full-resolution developed image (after geometry and crop)."""
        p = self._params(params)
        plan = self._plan(p, apply_crop)
        if plan.identity:
            return self.oriented_size()
        return plan.final_size()

    # -------------------------------------------------- geometry (stage A)
    @staticmethod
    def _rot_flip(img: np.ndarray, plan: _geo.GeoPlan) -> np.ndarray:
        r = np.rot90(img, -plan.k90) if plan.k90 else img
        if plan.flip_h:
            r = r[:, ::-1]
        if plan.flip_v:
            r = r[::-1]
        return r

    def _geometry(self, src: np.ndarray, plan: _geo.GeoPlan, ss: float) -> tuple[np.ndarray, float]:
        """src (base orientation, ``ss`` px per full-res px) → final frame at the same scale."""
        if plan.identity:
            return src, ss
        if not plan.warp:
            r = self._rot_flip(src, plan)
            h, w = r.shape[:2]
            x0, y0, x1, y1 = plan.crop
            c0 = min(w - 1, max(0, int(round(x0 * ss))))
            c1 = max(c0 + 1, min(w, int(round(x1 * ss))))
            r0 = min(h - 1, max(0, int(round(y0 * ss))))
            r1 = max(r0 + 1, min(h, int(round(y1 * ss))))
            return np.ascontiguousarray(r[r0:r1, c0:c1], dtype=np.float32), ss
        ow = max(1, int(round(plan.final_w * ss)))
        oh = max(1, int(round(plan.final_h * ss)))
        return _geo.warp(src, plan, 0.0, 0.0, ow, oh, ss, ss), ss

    # ---------------------------------------------------- scene (stage B)
    def _airlight(self, p: DevParams) -> np.ndarray:
        """Global atmospheric light from a small render (same for preview, export and regions)."""
        wb = self._wb3(p)
        m = self._linear_matrix(p)
        key = (tuple(np.round(wb, 9)), tuple(np.round(m, 9).ravel()))
        hit = getattr(self, "_air", None)
        if hit is not None and hit[0] == key:
            return hit[1]
        h, w = self.norm.shape
        f = max(1, -(-max(h, w) // 2 // 256))
        lv = self._level(f)
        x = np.minimum(lv.reshape(-1, 3), (1.0 / wb).astype(np.float32)) * wb.astype(np.float32)
        lin = (x @ m.T.astype(np.float32)).reshape(lv.shape)
        a = _tone.dehaze_airlight(lin)
        self._air = (key, a)
        return a

    def _scene(self, img: np.ndarray, p: DevParams, *, prewb: bool, wb3: np.ndarray, sat: np.ndarray | None,
               scale: float, long_side: float, draft: bool, airlight: np.ndarray | None = None) -> np.ndarray:
        """Stage B: highlights + WB (if ``prewb``) → colour matrix → defringe → dehaze → local tone."""
        m = self._linear_matrix(p)
        H, W, _ = img.shape
        flat_in = img.reshape(-1, 3)
        lin = np.empty((H, W, 3), dtype=np.float32)
        flat = lin.reshape(-1, 3)
        mode = p.highlight_mode
        if prewb and mode == "clip":
            thr = (1.0 / wb3).astype(np.float32)
            mT = np.ascontiguousarray((m * wb3[None, :]).T, dtype=np.float32)

            def k(s: int, e: int) -> None:
                np.matmul(np.minimum(flat_in[s:e], thr), mT, out=flat[s:e])

            run_chunks(k, flat.shape[0])
        else:
            if prewb:
                x = img * wb3.astype(np.float32)
                satl = wb3.astype(np.float32)
            else:
                x = img if mode == "clip" else img.copy()
                satl = sat if sat is not None else np.ones(3, dtype=np.float32)
            if mode != "clip":
                _wb.handle_highlights(x, satl, mode)
            mT = np.ascontiguousarray(m.T, dtype=np.float32)
            xf = x.reshape(-1, 3)

            def k(s: int, e: int) -> None:
                np.matmul(xf[s:e], mT, out=flat[s:e])

            run_chunks(k, flat.shape[0])
        if p.defringe and not draft:  # edge micro-detail: skipped while dragging
            _color.defringe(lin, p.defringe, scale)
        if p.dehaze:
            _tone.dehaze(lin, p.dehaze, airlight if airlight is not None else self._airlight(p))
        _tone.local_tone(lin, p, scale, long_side, draft)
        return lin

    # ------------------------------------------------- encoding (stage C)
    def _encode_lut(self, p: DevParams) -> np.ndarray:
        """(1 or 3, 65536) float32: contrast → output curve → parametric → point curves."""
        key = _p_key(p, ("contrast", "gamma_mode", "gamma", "curve_p_highlights", "curve_p_lights", "curve_p_darks",
                         "curve_p_shadows", "curve_master", "curve_red", "curve_green", "curve_blue"))
        st = self._preview_cache.setdefault(-1, {})
        if st.get("enc_key") == key:
            return st["enc"]
        scale = np.float32(LUT_SIZE - 1)
        xs = (np.arange(LUT_SIZE, dtype=np.float32) + 0.5) / scale
        base = transfer(_contrast_curve(np.minimum(xs, 1.0), p.contrast), p.gamma_mode, p.gamma)
        tabs = _tone.tone_tables(p)
        if tabs is None:
            lut = base[None, :]
        else:
            grid = np.linspace(0.0, 1.0, tabs.shape[1], dtype=np.float32)
            lut = np.stack([np.interp(base, grid, tabs[c]).astype(np.float32) for c in range(3)])
        st["enc_key"], st["enc"] = key, lut
        return lut

    def _encode(self, lin: np.ndarray, p: DevParams, exact: bool) -> np.ndarray:
        """Stage C: per-channel tone curves (+ output curve), HSL/vibrance, grading."""
        H, W, _ = lin.shape
        if exact:
            enc = transfer(_contrast_curve(lin, p.contrast), p.gamma_mode, p.gamma)
            tabs = _tone.tone_tables(p)
            if tabs is not None:
                grid = np.linspace(0.0, 1.0, tabs.shape[1], dtype=np.float32)
                for c in range(3):
                    enc[..., c] = np.interp(enc[..., c], grid, tabs[c])
        else:
            lut = self._encode_lut(p)
            src = lin.reshape(-1, 3)
            enc = np.empty((H, W, 3), dtype=np.float32)
            ef = enc.reshape(-1, 3)
            sc = np.float32(LUT_SIZE - 1)
            if lut.shape[0] == 1:
                l0 = lut[0]
                offs = None
            else:
                l0 = np.ascontiguousarray(lut).reshape(-1)
                offs = (np.arange(3, dtype=np.int32) * LUT_SIZE)[None, :]

            def k(s: int, e: int) -> None:
                v = src[s:e] * sc
                np.maximum(v, np.float32(0.0), out=v)
                np.minimum(v, sc, out=v)
                ii = v.astype(np.int32)
                if offs is not None:
                    ii += offs
                np.take(l0, ii, out=ef[s:e])

            run_chunks(k, src.shape[0])
        flat = enc.reshape(-1, 3)
        _hsl.apply_hsl(flat, _hsl.hsl_tables(p), float(p.vibrance))
        _grading.apply_grading(flat, _grading.grading_table(p))
        return enc

    # ----------------------------------------------------- detail (stage D)
    @staticmethod
    def _detail(enc: np.ndarray, p: DevParams, scale: float, draft: bool) -> np.ndarray:
        if draft:
            return enc
        out = _detail.noise_reduction(enc, p, scale)
        _detail.sharpen(out, p, scale)
        return out

    # ---------------------------------------------------- effects (stage E)
    def _effects(self, img: np.ndarray, p: DevParams, frame: tuple[float, float], x0: float, y0: float,
                 s_out: float, cache: dict | None = None) -> None:
        H, W, _ = img.shape
        if p.vignette_amount:
            key = ("vig", frame, x0, y0, W, H, s_out) + _p_key(p, ("vignette_midpoint", "vignette_roundness",
                                                                  "vignette_feather"))
            mask = cache.get(key) if cache is not None else None
            if mask is None:
                mask = _fx.vignette_mask(p, frame[0], frame[1], x0, y0, W, H, s_out)
                if cache is not None:
                    cache.clear() if len(cache) > 8 else None
                    cache[key] = mask
            _fx.apply_vignette(img, mask, float(p.vignette_amount), float(p.vignette_highlights))
        if p.grain_amount > 0:
            key = ("grain", x0, y0, W, H, s_out) + _p_key(p, ("grain_size", "grain_roughness"))
            fld = cache.get(key) if cache is not None else None
            if fld is None:
                fld = _fx.grain_field(p, self.seed, x0, y0, W, H, s_out)
                if cache is not None:
                    cache[key] = fld
            _fx.apply_grain(img, fld, float(p.grain_amount))

    # --------------------------------------------------------------- API
    def develop(
        self,
        params: DevParams | Mapping[str, Any] | None = None,
        *,
        demosaic: str | None = None,
        half_size: bool = False,
        apply_crop: bool = True,
        orient: bool = True,
    ) -> np.ndarray:
        """Full-quality development → float32 H×W×3 in [0, 1] (output curve applied)."""
        p = self._params(params)
        mode = p.highlight_mode
        mul = self.wb_multipliers(p)
        wb3 = np.array([mul[0], 0.5 * (mul[1] + mul[3]), mul[2]], dtype=np.float32)
        cfa = self._wb_cfa(p, clip=(mode == "clip"))
        if half_size:
            rgb = self._superpixel(cfa)
            ss = 0.5
        else:
            rgb = _dm.demosaic(cfa, self.pattern, demosaic or p.demosaic)
            ss = 1.0
        if orient:
            rgb = apply_orientation(rgb, self.orientation)
        h, w = rgb.shape[:2]
        base = (int(round(w / ss)), int(round(h / ss))) if half_size else (w, h)
        if half_size:
            base = (self.oriented_size() if orient else (self.norm.shape[1], self.norm.shape[0]))
        plan = self._plan(p, apply_crop, base)
        feats = self._features(p)
        if plan.identity and not feats and mode == "clip":
            # DEV v1 maths, bit-exact
            m = self._linear_matrix(p).astype(np.float32)
            lin = (rgb.reshape(-1, 3) @ m.T).reshape(h, w, 3)
            return transfer(_contrast_curve(lin, p.contrast), p.gamma_mode, p.gamma)
        img, _ = self._geometry(rgb, plan, ss)
        del rgb, cfa
        out = self._run_tail(img, p, prewb=False, wb3=wb3, sat=wb3, scale=ss,
                             frame=(plan.final_w if not plan.identity else base[0],
                                    plan.final_h if not plan.identity else base[1]),
                             x0=0.0, y0=0.0, draft=False, exact=True, cache=None)
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _run_tail(self, img: np.ndarray, p: DevParams, *, prewb: bool, wb3: np.ndarray, sat: np.ndarray | None,
                  scale: float, frame: tuple[float, float], x0: float, y0: float, draft: bool, exact: bool,
                  cache: dict | None, long_side: float | None = None) -> np.ndarray:
        """Stages B–E on a geometry-resolved image (no caching)."""
        H, W, _ = img.shape
        ls = long_side if long_side is not None else max(H, W)
        lin = self._scene(img, p, prewb=prewb, wb3=np.asarray(wb3, dtype=np.float64), sat=sat, scale=scale,
                          long_side=ls, draft=draft)
        enc = self._encode(lin, p, exact)
        del lin
        out = self._detail(enc, p, scale, draft)
        self._effects(out, p, frame, x0, y0, scale, cache)
        return out

    # ------------------------------------------------------------ preview
    def _preview_state(self, max_side: int) -> dict[str, Any]:
        st = self._preview_cache.get(max_side)
        if st is not None:
            return st
        h2, w2 = self.norm.shape[0] // 2, self.norm.shape[1] // 2
        f = max(1, -(-max(h2, w2) // max_side))  # ceil
        sp = self._level(f)
        n = sp.shape[0] * sp.shape[1]
        st = {
            "src": sp.reshape(n, 3),
            "shape": sp.shape,
            "a": np.empty((n, 3), dtype=np.float32),
            "b": np.empty((n, 3), dtype=np.float32),
            "idx": np.empty((n, 3), dtype=np.uint16),
            "y": np.empty(n, dtype=np.float32),
            "lut_key": None,
            "lut": None,
            "gain_key": None,
            "gain": None,
            "bounds": [(s, min(s + _CHUNK, n)) for s in range(0, n, _CHUNK)],
        }
        self._preview_cache[max_side] = st
        return st

    def preview(
        self,
        params: DevParams | Mapping[str, Any] | None = None,
        max_side: int = 1600,
        *,
        apply_crop: bool = True,
        draft: bool = False,
    ) -> np.ndarray:
        """Fast uint8 preview (half-size superpixel pyramid, cached).

        ``apply_crop=False`` returns the image with its geometry but without the
        crop (for editing the crop). ``draft=True`` renders at half resolution
        without NR/sharpening and upsamples (slider drags).
        """
        p = self._params(params)
        with self._preview_lock:
            plan = self._plan(p, apply_crop)
            if plan.identity and not self._features(p) and p.highlight_mode == "clip":
                return self._preview_locked(p, int(max_side))
            return self._preview_general(p, int(max_side), plan, apply_crop, bool(draft))

    def _preview_locked(self, p: DevParams, max_side: int) -> np.ndarray:
        """DEV v1 fast path: WB-clip → one 3×3 matrix → fused contrast+transfer+quantization LUT."""
        st = self._preview_state(max_side)
        scale = np.float32(LUT_SIZE - 1)
        wb = self._wb3(p)
        # WB then clip at 1  ≡  clip at 1/wb then WB (WB folded into the matrix).
        thr = (1.0 / wb).astype(np.float32)
        hl_sh = bool(p.highlights or p.shadows)
        m = self._linear_matrix(p) * wb[None, :] * (1.0 if hl_sh else float(scale))
        mT = np.ascontiguousarray(m.T, dtype=np.float32)
        wy = (m.T @ REC709.astype(np.float64) * (GAIN_LUT_SIZE - 1)).astype(np.float32)
        gmax = np.float32(GAIN_LUT_SIZE - 1)
        gain = None
        if hl_sh:
            key = (p.highlights, p.shadows)
            if st["gain_key"] != key:
                yy = np.linspace(0.0, 1.0, GAIN_LUT_SIZE, dtype=np.float32)
                st["gain"] = (_hl_sh_gain(yy, p.highlights, p.shadows) * scale).astype(np.float32)
                st["gain_key"] = key
            gain = st["gain"]
        key = (p.contrast, p.gamma_mode, p.gamma)
        if st["lut_key"] != key:
            xs = (np.arange(LUT_SIZE, dtype=np.float32) + 0.5) / scale
            st["lut"] = to_uint8(transfer(_contrast_curve(np.minimum(xs, 1.0), p.contrast), p.gamma_mode, p.gamma))
            st["lut_key"] = key
        lut = st["lut"]
        src, a, b, idx, yb = st["src"], st["a"], st["b"], st["idx"], st["y"]
        out = np.empty((src.shape[0], 3), dtype=np.uint8)  # fresh: caller owns the result

        def run(s: int, e: int) -> None:
            aa, bb, ii = a[s:e], b[s:e], idx[s:e]
            ss = src[s:e]
            for c in range(3):
                np.minimum(ss[:, c], thr[c], out=aa[:, c])
            np.matmul(aa, mT, out=bb)
            if gain is not None:
                y = yb[s:e]
                np.matmul(aa, wy, out=y)
                np.clip(y, 0.0, gmax, out=y)
                g = gain[y.astype(np.uint16)]
                for c in range(3):
                    col = bb[:, c]
                    np.multiply(col, g, out=col)
            np.clip(bb, 0.0, scale, out=bb)
            np.copyto(ii, bb, casting="unsafe")
            out[s:e] = lut[ii]

        run_chunks(run, src.shape[0], _CHUNK)
        return out.reshape(st["shape"])

    def _preview_level(self, plan: _geo.GeoPlan, max_side: int) -> int:
        """Superpixel level f (box factor) whose resolution fits ``max_side`` for this geometry."""
        h2, w2 = self.norm.shape[0] // 2, self.norm.shape[1] // 2
        if plan.identity:
            return max(1, -(-max(h2, w2) // max_side))
        long_full = max(plan.final_w, plan.final_h)
        return max(1, int(math.ceil(long_full * 0.5 / max_side - 1e-9)))

    def _preview_general(self, p: DevParams, max_side: int, plan: _geo.GeoPlan, apply_crop: bool,
                         draft: bool) -> np.ndarray:
        f = self._preview_level(plan, max_side)
        fw = f
        if draft:
            # draft: ≤ ~0.25 Mpx working resolution (integer box factor), upsampled at the end
            h2, w2 = self.norm.shape[0] // 2, self.norm.shape[1] // 2
            npx = (h2 // f) * (w2 // f) if plan.identity else (plan.final_w * plan.final_h) / (4.0 * f * f)
            k = max(2, int(math.ceil(math.sqrt(npx / DRAFT_PIXELS))))
            fw = f * k
            if min(h2 // fw, w2 // fw) < 8:
                fw = f
        slot_key = (max_side, apply_crop, draft)
        slots = self._stages.get(slot_key)
        if slots is None:
            slots = {}
            self._stages[slot_key] = slots
            while len(self._stages) > 3:
                self._stages.popitem(last=False)
        else:
            self._stages.move_to_end(slot_key)
        ss = 0.5 / fw
        # ---- A: geometry
        ka = (fw, plan.key)
        hit = slots.get("A")
        if hit is not None and hit[0] == ka:
            imgA = hit[1]
        else:
            imgA, _ = self._geometry(self._level(fw), plan, ss)
            slots["A"] = (ka, imgA)
            for s in ("B", "C", "D", "E"):
                slots.pop(s, None)
        H, W, _ = imgA.shape
        wb3 = self._wb3(p)
        # ---- B: scene
        kb = (ka, tuple(np.round(wb3, 9))) + _p_key(p, _B_KEYS)
        hit = slots.get("B")
        if hit is not None and hit[0] == kb:
            lin = hit[1]
        else:
            air = self._airlight(p) if p.dehaze else None
            lin = self._scene(imgA, p, prewb=True, wb3=wb3, sat=None, scale=ss, long_side=max(H, W),
                              draft=draft, airlight=air)
            slots["B"] = (kb, lin)
            for s in ("C", "D", "E"):
                slots.pop(s, None)
        # ---- C: encoding / colour
        kc = (kb,) + _p_key(p, _C_KEYS)
        hit = slots.get("C")
        if hit is not None and hit[0] == kc:
            enc = hit[1]
        else:
            enc = self._encode(lin, p, exact=False)
            slots["C"] = (kc, enc)
            for s in ("D", "E"):
                slots.pop(s, None)
        # ---- D: detail
        kd = (kc,) + (() if draft else _p_key(p, _D_KEYS))
        hit = slots.get("D")
        if hit is not None and hit[0] == kd:
            det = hit[1]
        else:
            feats = self._features(p)
            if draft or not (feats & {"nr", "sharpen"}):
                det = enc
            else:
                det = self._detail(enc.copy(), p, ss, False)
            slots["D"] = (kd, det)
            slots.pop("E", None)
        # ---- E: effects + quantization
        ke = (kd,) + _p_key(p, _E_KEYS)
        hit = slots.get("E")
        if hit is not None and hit[0] == ke:
            return hit[1].copy()
        frame = (plan.final_w, plan.final_h) if not plan.identity else tuple(map(float, self.oriented_size()))
        if p.vignette_amount or p.grain_amount > 0:
            img = det.copy()
            fxc = slots.setdefault("fx", {})
            self._effects(img, p, frame, 0.0, 0.0, ss, fxc)
        else:
            img = det
        out = quantize8(img)
        if fw != f:
            out = self._upsample(out, plan, max_side, f)
        slots["E"] = (ke, out)
        return out.copy()

    def _upsample(self, img: np.ndarray, plan: _geo.GeoPlan, max_side: int, f: int) -> np.ndarray:
        from PIL import Image

        if plan.identity:
            th, tw = self._level(f).shape[:2]
        elif not plan.warp:
            th, tw = self._geometry(self._level(f), plan, 0.5 / f)[0].shape[:2]
        else:
            ss = 0.5 / f
            th, tw = max(1, int(round(plan.final_h * ss))), max(1, int(round(plan.final_w * ss)))
        return np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((tw, th), Image.BILINEAR))

    # ------------------------------------------------------- zoom regions
    def render_region(
        self,
        params: DevParams | Mapping[str, Any] | None,
        rect: tuple[float, float, float, float],
        out_w: int,
        out_h: int,
        *,
        demosaic: str | None = None,
    ) -> np.ndarray:
        """Develops ``rect`` = (x0, y0, x1, y1), normalized on the FINAL image, at out_w × out_h (uint8).

        Only the needed part of the mosaic is demosaiced (full resolution when the
        requested scale exceeds the superpixel's); geometry is inverse-mapped.
        """
        p = self._params(params)
        out_w, out_h = max(1, int(out_w)), max(1, int(out_h))
        x0n, y0n, x1n, y1n = (float(np.clip(v, 0.0, 1.0)) for v in rect)
        if x1n - x0n < 1e-6:
            x1n = min(1.0, x0n + 1e-3)
        if y1n - y0n < 1e-6:
            y1n = min(1.0, y0n + 1e-3)
        with self._preview_lock:
            plan = self._plan(p, True)
        bw, bh = self.oriented_size()
        if plan.identity:
            plan = _geo.make_plan(p, bw, bh, True)
        fw_, fh_ = plan.final_w, plan.final_h
        s = out_w / ((x1n - x0n) * fw_)
        X0 = x0n * fw_
        Y0 = 0.5 * (y0n + y1n) * fh_ - 0.5 * out_h / s
        feats = self._features(p)
        long_side = max(fw_, fh_) * s
        pad = 0
        if feats & {"localtone", "dehaze"}:
            pad = int(math.ceil(0.05 * long_side)) + 8
        elif feats & {"nr", "sharpen", "defringe"}:
            pad = 16
        px0 = max(0, int(math.floor(X0 * s)) - pad) / s if pad else X0
        py0 = max(0.0, Y0 - pad / s) if pad else Y0
        pw = out_w + int(round((X0 - px0) * s)) + pad
        ph = out_h + int(round((Y0 - py0) * s)) + pad
        pw = min(pw, int(math.ceil((fw_ - px0) * s)) + 1) if pad else pw
        ph = min(ph, int(math.ceil((fh_ - py0) * s)) + 1) if pad else ph
        pw, ph = max(pw, out_w + int(round((X0 - px0) * s))), max(ph, out_h + int(round((Y0 - py0) * s)))
        wb3 = self._wb3(p)
        if s <= 0.5:
            f = max(1, int(math.floor(0.5 / s + 1e-9)))
            src = self._level(f)
            ss = 0.5 / f
            img = _geo.warp(src, plan, px0, py0, pw, ph, s, ss)
            prewb, sat = True, None
        else:
            # bounding box of the needed base area (all three CA channels)
            t = np.linspace(0.0, 1.0, 17)
            bx = np.concatenate([px0 + t * pw / s, px0 + t * pw / s, np.full(17, px0), np.full(17, px0 + pw / s)])
            by = np.concatenate([np.full(17, py0), np.full(17, py0 + ph / s), py0 + t * ph / s, py0 + t * ph / s])
            xs, ys = [], []
            for c in range(3):
                x, y = plan.final_to_base(bx, by, c)
                xs.append(x)
                ys.append(y)
            xa, ya = np.concatenate(xs), np.concatenate(ys)
            m = 8
            bx0 = int(max(0, math.floor(xa.min()) - m))
            by0 = int(max(0, math.floor(ya.min()) - m))
            bx1 = int(min(bw, math.ceil(xa.max()) + m))
            by1 = int(min(bh, math.ceil(ya.max()) + m))
            if bx1 <= bx0 or by1 <= by0:
                return np.zeros((out_h, out_w, 3), dtype=np.uint8)
            src, ox, oy = self._demosaic_rect(p, bx0, by0, bx1, by1, demosaic)
            img = _geo.warp(src, plan, px0, py0, pw, ph, s, 1.0, (ox, oy))
            prewb = False
            sat = wb3.astype(np.float32)
            if p.highlight_mode == "clip":
                sat = None
        air = self._airlight(p) if p.dehaze else None
        lin = self._scene(img, p, prewb=prewb, wb3=wb3, sat=sat, scale=s, long_side=long_side, draft=False,
                          airlight=air)
        enc = self._encode(lin, p, exact=False)
        del lin
        enc = self._detail(enc, p, s, False)
        self._effects(enc, p, (fw_, fh_), px0, py0, s, None)
        ox = int(round((X0 - px0) * s))
        oy = int(round((Y0 - py0) * s))
        res = enc[oy : oy + out_h, ox : ox + out_w]
        if res.shape[:2] != (out_h, out_w):
            res = np.pad(res, ((0, out_h - res.shape[0]), (0, out_w - res.shape[1]), (0, 0)))
        return quantize8(res)

    def _demosaic_rect(self, p: DevParams, x0: int, y0: int, x1: int, y1: int,
                       method: str | None) -> tuple[np.ndarray, float, float]:
        """WB'd, demosaiced, EXIF-oriented crop covering base coords [x0, x1) × [y0, y1).

        Returns (rgb, ox, oy): the array and the base coords of its top-left corner.
        """
        bw, bh = self.oriented_size()
        cx = np.array([x0, x1, x0, x1], dtype=np.float64)
        cy = np.array([y0, y0, y1, y1], dtype=np.float64)
        sx, sy = _display_to_stored(self.orientation, cx, cy, float(bw), float(bh))
        H, W = self.norm.shape
        m = 8
        sx0 = max(0, int(math.floor(sx.min())) - m) & ~1
        sy0 = max(0, int(math.floor(sy.min())) - m) & ~1
        sx1 = min(W, int(math.ceil(sx.max())) + m)
        sy1 = min(H, int(math.ceil(sy.max())) + m)
        sub = self.norm[sy0:sy1, sx0:sx1]
        site = self._site_multipliers(self.wb_multipliers(p))  # even origin → same CFA phase
        h, w = sub.shape
        cfa = sub * np.tile(site, ((h + 1) // 2, (w + 1) // 2))[:h, :w]
        if p.highlight_mode == "clip":
            np.minimum(cfa, 1.0, out=cfa)
        if h < 2 or w < 2:
            rgb = np.repeat(cfa[..., None], 3, axis=2).astype(np.float32)
        else:
            rgb = _dm.demosaic(cfa, self.pattern, method or p.demosaic)
        rgb = apply_orientation(rgb, self.orientation)
        ex, ey = self._stored_to_display(np.array([sx0, sx1], dtype=np.float64),
                                         np.array([sy0, sy1], dtype=np.float64))
        return rgb, float(np.min(ex)), float(np.min(ey))

    def _stored_to_display(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward of :func:`apply_orientation` for points (stored → display coords)."""
        H, W = self.norm.shape
        o = self.orientation
        if o == 2:
            return W - x, y
        if o == 3:
            return W - x, H - y
        if o == 4:
            return x, H - y
        if o == 5:
            return y, x
        if o == 6:
            return H - y, x
        if o == 7:
            return H - y, W - x
        if o == 8:
            return y, W - x
        return x, y

    # ---------------------------------------------------------- analysis
    histogram = staticmethod(histogram)
    clipping_mask = staticmethod(clipping_mask)

    def _final_to_base_point(self, p: DevParams, x: float, y: float) -> tuple[float, float]:
        plan = self._plan(p, True)
        bw, bh = self.oriented_size()
        if plan.identity:
            return x * bw, y * bh
        bx, by = plan.final_to_base(np.array([x * plan.final_w]), np.array([y * plan.final_h]), 1)
        return float(bx[0]), float(by[0])

    def pixel_values(self, x: float, y: float,
                     params: DevParams | Mapping[str, Any] | None = None) -> tuple[int, int, int]:
        """RGB (0–255) of the final image at normalized (x, y), from the (cached) 1600 px preview."""
        img = self.preview(params, 1600)
        h, w = img.shape[:2]
        i = min(h - 1, max(0, int(float(y) * h)))
        j = min(w - 1, max(0, int(float(x) * w)))
        r, g, b = (int(v) for v in img[i, j])
        return r, g, b

    def sample_wb(self, x: float, y: float, radius: int = 3, *,
                  params: DevParams | Mapping[str, Any] | None = None) -> list[float]:
        """Custom WB multipliers [R, G, B] (G = 1) that neutralize the patch at normalized final (x, y).

        The point is inverse-mapped through the geometry and the EXIF orientation to
        the sensor; the patch is the CFA-aligned window of ``2·radius`` sensor pixels
        around it (``radius`` superpixels across), averaged per colour in pre-WB
        camera space. 2×2 blocks with a (nearly) saturated photosite are ignored when
        unsaturated ones remain.
        """
        p = self._params(params)
        with self._preview_lock:
            bx, by = self._final_to_base_point(p, float(x), float(y))
        bw, bh = self.oriented_size()
        ix = min(bw - 1, max(0, int(math.floor(bx))))
        iy = min(bh - 1, max(0, int(math.floor(by))))
        sx, sy = _display_to_stored(self.orientation, np.array([ix + 0.5]), np.array([iy + 0.5]), float(bw), float(bh))
        H, W = self.norm.shape
        r = max(1, int(radius))
        xs, ys = int(math.floor(sx[0])), int(math.floor(sy[0]))
        y0 = max(0, min(H - 2 * r, ys - r)) & ~1
        x0 = max(0, min(W - 2 * r, xs - r)) & ~1
        patch = self.norm[y0 : y0 + 2 * r, x0 : x0 + 2 * r]
        h2, w2 = patch.shape[0] // 2, patch.shape[1] // 2
        if h2 == 0 or w2 == 0:
            return [float(v) for v in self._wb3(p)]
        blocks = patch[: 2 * h2, : 2 * w2].reshape(h2, 2, w2, 2)
        ok = blocks.max(axis=(1, 3)) < 0.98
        acc = np.zeros(3)
        cnt = np.zeros(3)
        for i, c in enumerate(self.pattern):
            v = patch[i >> 1 : 2 * h2 : 2, i & 1 : 2 * w2 : 2]
            if ok.any() and not ok.all():
                v = v[ok]
            acc[c] += float(np.mean(v))
            cnt[c] += 1
        mean = np.maximum(acc / np.maximum(cnt, 1), 1e-6)
        mul = np.clip(mean[1] / mean, 1e-3, 64.0)
        return [float(mul[0]), 1.0, float(mul[2])]

    def auto_wb(self) -> dict[str, Any]:
        """{"wb_mode": "custom", "wb_custom": [r, 1, b]} — robust grey world (see ``estimate_auto_wb``).

        Neutral only with ``temperature_shift``/``tint_shift`` at 0.
        """
        h, w = self.norm.shape
        f = max(1, -(-max(h, w) // 2 // 512))
        with self._preview_lock:
            lv = self._level(f)
        est = _wb.estimate_auto_wb(lv)
        if est is None:
            r, g, b = (float(v) for v in self.wb_multipliers({"wb_mode": "as_shot"})[:3])
            est = (r / g, 1.0, b / g)
        return {"wb_mode": "custom", "wb_custom": [round(float(v), 4) for v in est]}

    def as_shot_kelvin(self) -> tuple[float, float]:
        """(kelvin, tint) of the as-shot WB (cam_xyz if COLR flags bit1, otherwise approximated)."""
        return multipliers_to_kelvin(list(self.color.as_shot_wb)[:3], self.color)

    def auto_tone(self, params: DevParams | Mapping[str, Any] | None = None) -> dict[str, Any]:
        """exposure_ev, contrast, highlights, shadows, whites, blacks (+ vibrance) from luminance percentiles."""
        p = self._params(params)
        h, w = self.norm.shape
        f = max(1, -(-max(h, w) // 2 // 400))
        with self._preview_lock:
            lv = self._level(f)
        base = p.copy()
        for k in ("exposure_ev", "highlights", "shadows", "whites", "blacks", "clarity", "texture", "dehaze",
                  "cal_shadows_tint"):
            setattr(base, k, 0.0)
        wb3 = self._wb3(base)
        ls = max(lv.shape[:2])
        lin0 = self._scene(lv, base, prewb=True, wb3=wb3, sat=None, scale=0.5 / f, long_side=ls, draft=True)
        y0 = luminance(lin0)

        def render(d: dict) -> np.ndarray:
            q = base.copy()
            for k, v in d.items():
                setattr(q, k, float(v))
            return self._scene(lv, q, prewb=True, wb3=wb3, sat=None, scale=0.5 / f, long_side=ls, draft=True)

        enc = srgb_encode(np.clip(lin0, 0.0, 1.0))
        mx, mn = enc.max(axis=2), enc.min(axis=2)
        sat = np.where(mx > 1e-3, (mx - mn) / np.maximum(mx, 1e-3), 0.0)
        return _auto_tone(y0, render, float(np.mean(sat)))

    # ------------------------------------------------------------ export
    def export(
        self,
        path: str | os.PathLike,
        params: DevParams | Mapping[str, Any] | None = None,
        bits: int = 8,
        quality: int = 95,
        *,
        long_edge: int | None = None,
        output_sharpen: str = "none",
        embed_icc: bool = True,
        metadata: bool = True,
        demosaic: str | None = None,
        half_size: bool = False,
    ) -> Path:
        """Writes JPEG (8-bit), PNG (8/16) or TIFF (8/16): full pipeline, optional Lanczos resize to
        ``long_edge``, output sharpening, embedded sRGB ICC and basic EXIF (JPEG/PNG)."""
        path = Path(path)
        ext = path.suffix.lower()
        if bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")
        if ext in (".jpg", ".jpeg") and bits != 8:
            raise ValueError("JPEG only supports 8 bits")
        if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            raise ValueError(f"unsupported export format {ext!r}")
        if output_sharpen not in OUTPUT_SHARPEN:
            raise ValueError(f"output_sharpen must be one of {sorted(OUTPUT_SHARPEN)}")
        img = self.develop(params, demosaic=demosaic, half_size=half_size)
        if long_edge:
            img = _resize_lanczos(img, int(long_edge))
        amt = OUTPUT_SHARPEN[output_sharpen]
        if amt:
            sp = SimpleNamespace(sharpen_amount=0.0, sharpen_radius=0.6, sharpen_detail=0.5, sharpen_masking=0.0)
            _detail.sharpen(img, sp, 1.0, amount=amt, radius=0.6)
            np.clip(img, 0.0, 1.0, out=img)
        icc = srgb_icc() if embed_icc else b""
        if bits == 16:
            a16 = to_uint16(img)
            if ext == ".png":
                write_png16(path, a16, icc or None)
            else:
                write_tiff16(path, a16, icc or None)
            return path
        from PIL import Image

        im = Image.fromarray(np.ascontiguousarray(to_uint8(img)))
        kw: dict[str, Any] = {}
        if icc:
            kw["icc_profile"] = icc
        if metadata and ext in (".jpg", ".jpeg", ".png"):
            kw["exif"] = build_exif(self.frawfile.meta, width=im.size[0], height=im.size[1]).tobytes()
        if ext in (".jpg", ".jpeg"):
            im.save(path, "JPEG", quality=int(quality), subsampling=0, **kw)
        elif ext == ".png":
            im.save(path, "PNG", **kw)
        else:
            im.save(path, "TIFF", **{k: v for k, v in kw.items() if k != "exif"})
        return path


def _resize_lanczos(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Lanczos-3 resize (Pillow, per channel in float32 'F' mode) so that the long side = long_edge."""
    from PIL import Image

    h, w = img.shape[:2]
    long_edge = max(1, int(long_edge))
    if max(h, w) == long_edge:
        return img
    sc = long_edge / max(h, w)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    out = np.empty((nh, nw, 3), dtype=np.float32)
    for c in range(3):
        ch = Image.fromarray(np.ascontiguousarray(img[..., c], dtype=np.float32))  # mode "F"
        out[..., c] = np.asarray(ch.resize((nw, nh), Image.LANCZOS), dtype=np.float32)
    np.clip(out, 0.0, 1.0, out=out)
    return out
