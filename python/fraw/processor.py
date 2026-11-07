"""FabriRAW development engine (SPEC §5 reference pipeline).

Pipeline (both :meth:`FRAWProcessor.develop` and the fast
:meth:`FRAWProcessor.preview` implement the same maths):

1. Crop to the HEAD active area and normalize
   ``x = (raw − black[cfa]) / (white − black[cfa])`` clipped to [0, 1].
2. White balance per CFA colour (as_shot / daylight / custom, then
   temperature and tint shifts). Multipliers are normalized so the smallest
   is 1 and the result is clipped to 1 ("highlight clip"): a saturated photosite
   stays at 1.0 in every channel, so blown highlights come out neutral white
   instead of magenta.
3. Demosaic (``bilinear`` | ``ahd``), or a 2×2 superpixel for ``half_size``.
4. ``rgb = rgb_cam · rgb`` (identity if COLR.flags bit0 is clear).
5. Tone, in linear light: exposure ``2**ev`` → highlights/shadows →
   saturation → contrast.

   * Highlights/shadows: a per-pixel gain ``2**(shadows·S·w_s(Y) +
     highlights·H·w_h(Y))`` driven by Rec.709 luminance Y, with smooth
     (smoothstep) weights ``w_s = 1 − smoothstep(0, 0.5, Y)`` and
     ``w_h = smoothstep(0.25, 1, Y)``; S = 1.5 EV, H = 1 EV at ±1. A gain
     preserves hue/chroma ratios.
   * Saturation: ``Y + s·(rgb − Y)`` (Rec.709 weights). It is a linear map that
     keeps Y, so it commutes with the luminance-driven gain above; it is applied
     *before* the contrast curve so that contrast + transfer curve can be
     fused into a single 1-D LUT in the preview path.
   * Contrast: an S-curve in a perceptual domain ``p = x**(1/2.2)`` pivoting on
     the middle grey ``p0 = 0.18**(1/2.2)``. Below the pivot
     ``p0·(p/p0)**a``, above ``1 − (1−p0)·((1−p)/(1−p0))**a`` with
     ``a = 2**contrast`` (C¹-continuous, fixes 0, 0.18 and 1). Per channel.

6. Transfer curve (``srgb`` piecewise IEC 61966-2-1 | ``power`` 1/gamma |
   ``linear``), clip to [0, 1], quantization.

Orientation (EXIF 1–8, HEAD.orientation) is applied to the developed image,
the preview and exports.
"""

from __future__ import annotations

import os
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import demosaic as _dm
from .devparams import DevParams
from .format import FrawFile

__all__ = [
    "FRAWProcessor",
    "srgb_encode",
    "transfer",
    "apply_orientation",
    "to_uint8",
    "to_uint16",
    "write_png16",
    "write_tiff16",
    "REC709",
    "TEMPERATURE_K",
    "TINT_K",
]

REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
TEMPERATURE_K = 0.5  # R·exp(+k·t), B·exp(−k·t): t = ±1 → ×1.65 / ×0.61
TINT_K = 0.5  # G·exp(−k·tint): +1 → magenta, −1 → green (Lightroom convention)
SHADOWS_EV = 1.5
HIGHLIGHTS_EV = 1.0
CONTRAST_GAMMA = 2.2
MID_GREY = 0.18
GAIN_LUT_SIZE = 4096  # highlights/shadows gain vs. luminance (smooth → small LUT)
LUT_SIZE = 65536  # linear-domain LUT; 4096 was ±6 levels off in the shadows of power curves
_CHUNK = 32768  # preview pixels per work item (fits in L2 cache)
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor | None:
    """Shared worker pool for the preview (numpy releases the GIL)."""
    global _POOL
    n = min(8, os.cpu_count() or 1)
    if n <= 1:
        return None
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="fraw-preview")
    return _POOL


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
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


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hl_sh_gain(y: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    ev = np.float32(0)
    if shadows:
        ev = ev + np.float32(shadows * SHADOWS_EV) * (1.0 - _smoothstep(0.0, 0.5, y))
    if highlights:
        ev = ev + np.float32(highlights * HIGHLIGHTS_EV) * _smoothstep(0.25, 1.0, y)
    return np.exp2(ev).astype(np.float32)


def _contrast_curve(x: np.ndarray, contrast: float) -> np.ndarray:
    """S-curve in p = x**(1/2.2) around middle grey. x is clipped to [0, 1]."""
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


def _saturation_matrix(s: float) -> np.ndarray:
    return (s * np.eye(3) + (1.0 - s) * np.outer(np.ones(3), REC709)).astype(np.float64)


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


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(img: np.ndarray) -> np.ndarray:
    return (np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png16(path: str | os.PathLike, img16: np.ndarray) -> None:
    """16-bit/channel RGB PNG (color type 2, bit depth 16, filter 0), via zlib."""
    a = np.ascontiguousarray(img16, dtype=">u2")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected H×W×3 uint16")
    h, w, _ = a.shape
    rows = np.empty((h, 1 + w * 6), dtype=np.uint8)
    rows[:, 0] = 0
    rows[:, 1:] = a.view(np.uint8).reshape(h, w * 6)
    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr)
    data += _png_chunk(b"IDAT", zlib.compress(rows.tobytes(), 6)) + _png_chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(data)


def write_tiff16(path: str | os.PathLike, img16: np.ndarray) -> None:
    """Baseline little-endian, uncompressed, 16-bit/channel RGB TIFF (single strip)."""
    a = np.ascontiguousarray(img16, dtype="<u2")
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected H×W×3 uint16")
    h, w, _ = a.shape
    pix = a.tobytes()
    SHORT, LONG, RATIONAL = 3, 4, 5
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
    n = len(entries)
    ifd_off = 8
    ifd_size = 2 + n * 12 + 4
    extra_off = ifd_off + ifd_size
    extra = b""
    body = []
    # Out-of-line values first, then pixel data (word aligned).
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
            field = struct.pack("<I", offsets[tag])
        else:
            field = val.ljust(4, b"\0")
        body.append(struct.pack("<HHI", tag, typ, cnt) + field)
    data = b"II*\0" + struct.pack("<I", ifd_off) + struct.pack("<H", n) + b"".join(body) + b"\0\0\0\0" + extra + pix
    with open(path, "wb") as f:
        f.write(data)


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
        self._preview_lock = threading.Lock()  # preview buffers are reused between calls

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

    # ------------------------------------------------------------ stages
    def _wb_cfa(self, p: DevParams) -> np.ndarray:
        site = self._site_multipliers(self.wb_multipliers(p))
        h, w = self.norm.shape
        smap = np.tile(site, ((h + 1) // 2, (w + 1) // 2))[:h, :w]
        out = self.norm * smap
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

    def _linear_matrix(self, p: DevParams) -> np.ndarray:
        """saturation · 2**ev · rgb_cam (all linear; see module docstring)."""
        ev = float(np.clip(p.exposure_ev, -4.0, 4.0))
        return _saturation_matrix(p.saturation) @ (2.0**ev * self.color_matrix())

    def _tone(self, rgb: np.ndarray, p: DevParams) -> np.ndarray:
        """camera RGB (WB'd, demosaiced) → output-referred [0,1] float32."""
        m = self._linear_matrix(p).astype(np.float32)
        h, w, _ = rgb.shape
        lin = (rgb.reshape(-1, 3) @ m.T).reshape(h, w, 3)
        if p.highlights or p.shadows:
            y = lin @ REC709
            lin *= _hl_sh_gain(y, p.highlights, p.shadows)[..., None]
        lin = _contrast_curve(lin, p.contrast)
        return transfer(lin, p.gamma_mode, p.gamma)

    # --------------------------------------------------------------- API
    def develop(
        self,
        params: DevParams | Mapping[str, Any] | None = None,
        *,
        demosaic: str | None = None,
        half_size: bool = False,
        orient: bool = True,
    ) -> np.ndarray:
        """Full-quality development → float32 H×W×3 in [0, 1] (transfer curve applied)."""
        p = self._params(params)
        cfa = self._wb_cfa(p)
        if half_size:
            rgb = self._superpixel(cfa)
        else:
            rgb = _dm.demosaic(cfa, self.pattern, demosaic or p.demosaic)
        out = self._tone(rgb, p)
        return apply_orientation(out, self.orientation) if orient else out

    def _preview_state(self, max_side: int) -> dict[str, Any]:
        st = self._preview_cache.get(max_side)
        if st is not None:
            return st
        sp = self._superpixel(self.norm)  # pre-WB camera RGB, half size
        h, w, _ = sp.shape
        f = max(1, -(-max(h, w) // max_side))  # ceil
        if f > 1:  # area (box) downscale by an integer factor
            hh, ww = h // f, w // f
            sp = sp[: hh * f, : ww * f].reshape(hh, f, ww, f, 3).mean(axis=(1, 3), dtype=np.float32)
        sp = np.ascontiguousarray(apply_orientation(sp, self.orientation), dtype=np.float32)
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

    def preview(self, params: DevParams | Mapping[str, Any] | None = None, max_side: int = 1600) -> np.ndarray:
        """Fast uint8 preview (half-size superpixel, cached; LUT-based tone).

        The first call builds and caches the downscaled superpixel; later calls
        only run WB-clip → one 3×3 matrix (WB·rgb_cam·exposure·saturation) →
        optional luminance gain → fused contrast+transfer+quantization LUT.
        """
        p = self._params(params)
        with self._preview_lock:
            return self._preview_locked(p, int(max_side))

    def _preview_locked(self, p: DevParams, max_side: int) -> np.ndarray:
        st = self._preview_state(max_side)
        scale = np.float32(LUT_SIZE - 1)
        mul = self.wb_multipliers(p)
        wb = np.array([mul[0], 0.5 * (mul[1] + mul[3]), mul[2]], dtype=np.float64)
        # WB then clip at 1  ≡  clip at 1/wb then WB (WB folded into the matrix).
        thr = (1.0 / wb).astype(np.float32)
        hl_sh = bool(p.highlights or p.shadows)
        m = self._linear_matrix(p) * wb[None, :] * (1.0 if hl_sh else float(scale))
        mT = np.ascontiguousarray(m.T, dtype=np.float32)
        # Luminance straight from the clipped input, already in gain-LUT index units.
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

        def run(se: tuple[int, int]) -> None:
            s, e = se
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
                for c in range(3):  # per-channel: ~7x faster than broadcasting g[:, None]
                    col = bb[:, c]
                    np.multiply(col, g, out=col)
            np.clip(bb, 0.0, scale, out=bb)
            np.copyto(ii, bb, casting="unsafe")
            out[s:e] = lut[ii]

        bounds = st["bounds"]
        ex = _executor()
        if ex is None or len(bounds) == 1:
            for se in bounds:
                run(se)
        else:
            list(ex.map(run, bounds))
        return out.reshape(st["shape"])

    # ------------------------------------------------------------ export
    def export(
        self,
        path: str | os.PathLike,
        params: DevParams | Mapping[str, Any] | None = None,
        bits: int = 8,
        quality: int = 95,
        *,
        demosaic: str | None = None,
        half_size: bool = False,
    ) -> Path:
        """Writes JPEG (8-bit), PNG (8/16) or TIFF (8/16). Orientation applied."""
        path = Path(path)
        ext = path.suffix.lower()
        if bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")
        if ext in (".jpg", ".jpeg") and bits != 8:
            raise ValueError("JPEG only supports 8 bits")
        if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            raise ValueError(f"unsupported export format {ext!r}")
        img = self.develop(params, demosaic=demosaic, half_size=half_size)
        if bits == 16:
            a16 = to_uint16(img)
            if ext == ".png":
                write_png16(path, a16)
            else:
                write_tiff16(path, a16)
            return path
        from PIL import Image

        im = Image.fromarray(np.ascontiguousarray(to_uint8(img)))
        if ext in (".jpg", ".jpeg"):
            im.save(path, "JPEG", quality=int(quality), subsampling=0)
        elif ext == ".png":
            im.save(path, "PNG")
        else:
            im.save(path, "TIFF")
        return path
