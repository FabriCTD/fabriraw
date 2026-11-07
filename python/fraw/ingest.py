"""Camera RAW -> FabriRAW (.fraw) ingest, built on rawpy/LibRaw.

``rawpy`` (and Pillow, for thumbnails) are imported lazily inside the functions
that need them, so ``import fraw.ingest`` works without the optional extras and
the pure mapping helpers below can be unit-tested with a fake rawpy object.

Mapping decisions (see also the docstrings of each helper):

* DATA is ``raw.raw_image`` — the COMPLETE sensor mosaic including masked
  margins, bit-for-bit, no black subtraction, no interpolation.  If the full
  mosaic has an odd width/height the LAST column/row is dropped (SPEC §4.1
  requires even dimensions).  Dropping at the end never shifts the CFA phase.
* The visible area (``sizes.left_margin/top_margin/width/height``) becomes the
  HEAD crop, clipped to the (possibly trimmed) mosaic.  If it covers the whole
  mosaic the crop is written as all-zero ("full image").
* ``cfa_pattern`` is expressed at pixel (0,0) of the FULL mosaic.  LibRaw's
  ``COLOR()``/``filters`` are relative to the visible area; with odd margins the
  phase shifts.  We read the visible-relative 2x2 pattern and shift it by the
  margins (:func:`pattern_full_from_visible`), then cross-check against
  ``raw.raw_color(y, x)`` which rawpy documents as full-image coordinates.
* LibRaw colour indices 0..3 are translated through ``color_desc`` ("RGBG":
  index 3 = second green) into SPEC colours 0=R 1=G 2=B.
* ``black_level[i]`` = ``black_level_per_channel[color_index_at_position_i]``.
  rawpy's ``black_level_per_channel`` already includes LibRaw's common
  ``black`` part (it runs LibRaw's ``adjust_bl``); rawpy 0.28 exposes no
  separate scalar.  Spatial black patterns larger than 2x2 (``cblack[6+]``) are
  not exposed by rawpy and are therefore not represented.
* ``rgb_cam``: rawpy 0.28 does NOT expose LibRaw's computed ``rgb_cam``.
  ``color_matrix`` is LibRaw's ``cmatrix`` (an optional matrix embedded in some
  files; all-zero for most cameras).  We therefore compute rgb_cam from
  ``rgb_xyz_matrix`` (``cam_xyz``) exactly like dcraw/LibRaw ``cam_xyz_coeff``
  (cam_rgb = cam_xyz · xyz_rgb(sRGB D65), rows normalised to 1, inverted).  If
  ``cam_xyz`` is unusable we fall back to ``color_matrix[:, :3]`` when valid,
  otherwise identity with flags.bit0 = 0.
"""

from __future__ import annotations

import datetime as _dt
import io
import math
import os
import struct
import uuid
from typing import Any, Iterable, Sequence

import numpy as np

from .format import Color, FrawFile, Head, Thumbnail
from .fset import FrawSet

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SOFTWARE",
    "ingest",
    "ingest_to_file",
    "ingest_burst",
    "from_rawpy",
    "pattern_full_from_visible",
    "map_cfa_pattern",
    "black_levels_for_pattern",
    "flip_to_orientation",
    "effective_bits_for",
    "normalize_wb",
    "rgb_cam_from_cam_xyz",
    "is_valid_matrix",
    "even_geometry",
]

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".cr3", ".cr2", ".nef", ".nrw", ".arw", ".sr2", ".dng", ".raf", ".orf", ".rw2", ".pef"}
)
SOFTWARE = "CTD Photo Studio 0.1.0"
ILLUMINANT_D65 = 21
COLR_FLAG_RGB_CAM = 1
COLR_FLAG_CAM_XYZ = 2
THUMB_JPEG_QUALITY = 85

# sRGB (linear, D65) -> XYZ, as used by dcraw/LibRaw (xyz_rgb).
_XYZ_RGB = np.array(
    [
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227],
    ],
    dtype=np.float64,
)

# LibRaw sizes.flip -> EXIF Orientation.  flip bits: 1 = mirror X, 2 = mirror Y,
# 4 = transpose (dcraw).  Common values: 0 -> 1, 3 -> 3 (180°), 5 -> 8 (90° CCW),
# 6 -> 6 (90° CW).  Inverse of dcraw's "50132467"[exif & 7] table.
_FLIP_TO_EXIF = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5, 5: 8, 6: 6, 7: 7}

_LETTER_TO_CFA = {"R": 0, "G": 1, "B": 2}


# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------
def _import_rawpy() -> Any:
    try:
        import rawpy  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "fraw.ingest needs rawpy to read camera RAW files: pip install fabriraw[ingest]  (or: pip install rawpy)"
        ) from e
    return rawpy


def _import_pil() -> Any:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - depends on environment
        raise ImportError(
            "fraw.ingest needs Pillow to build thumbnails: pip install pillow  (or call ingest(..., thumbnail=False))"
        ) from e
    return Image


def _rawpy_thumb_errors() -> tuple[type[BaseException], ...]:
    try:
        import rawpy  # noqa: PLC0415
    except ImportError:
        return ()
    return (rawpy.LibRawError,)


# ---------------------------------------------------------------------------
# Pure mapping helpers (unit-tested with fakes)
# ---------------------------------------------------------------------------
def pattern_full_from_visible(vis: Sequence[Sequence[int]], top_margin: int, left_margin: int) -> list[list[int]]:
    """Re-express a 2x2 CFA pattern given at the visible origin at the full-mosaic origin.

    Full pixel (y, x) is visible pixel (y - top, x - left), hence
    ``full[y][x] = vis[(y - top) % 2][(x - left) % 2]``.  Only the parity of the
    margins matters.
    """
    v = [[int(vis[r][c]) for c in range(2)] for r in range(2)]
    return [[v[(y - top_margin) % 2][(x - left_margin) % 2] for x in range(2)] for y in range(2)]


def map_cfa_pattern(indices: Sequence[Sequence[int]], color_desc: bytes | str) -> tuple[int, int, int, int]:
    """LibRaw colour indices (2x2, 0..3) -> SPEC cfa_pattern (row-major, 0=R 1=G 2=B).

    ``color_desc`` is LibRaw's ``cdesc`` (e.g. b"RGBG": index 3 is the second
    green).  Raises ValueError unless the result is a genuine Bayer pattern
    (one R, one B, two G).
    """
    desc = color_desc.decode("ascii", "replace") if isinstance(color_desc, (bytes, bytearray)) else str(color_desc)
    desc = desc.rstrip("\x00")
    out: list[int] = []
    for r in range(2):
        for c in range(2):
            idx = int(indices[r][c])
            if not 0 <= idx < len(desc):
                raise ValueError(f"CFA colour index {idx} not described by color_desc {desc!r}")
            letter = desc[idx].upper()
            if letter not in _LETTER_TO_CFA:
                raise ValueError(f"non-RGB colour filter {letter!r} (color_desc {desc!r}); only RGB Bayer is supported")
            out.append(_LETTER_TO_CFA[letter])
    if sorted(out) != [0, 1, 1, 2]:
        raise ValueError(f"CFA {out} is not a 2x2 Bayer pattern (needs one R, two G, one B)")
    return (out[0], out[1], out[2], out[3])


def black_levels_for_pattern(
    indices: Sequence[Sequence[int]], black_per_channel: Sequence[float]
) -> tuple[int, int, int, int]:
    """Black level per CFA position from LibRaw's per-colour-index list (index 3 = G2)."""
    bl = [int(round(float(b))) for b in black_per_channel]
    if len(bl) < 4:
        bl = bl + [bl[1] if len(bl) > 1 else 0] * (4 - len(bl))
    out = []
    for r in range(2):
        for c in range(2):
            v = bl[int(indices[r][c])]
            if v < 0:
                raise ValueError(f"negative black level {v}")
            out.append(v)
    return (out[0], out[1], out[2], out[3])


def flip_to_orientation(flip: int) -> int:
    """LibRaw ``sizes.flip`` -> EXIF orientation (1..8). Unknown values -> 1."""
    try:
        return _FLIP_TO_EXIF.get(int(flip), 1)
    except (TypeError, ValueError):
        return 1


def effective_bits_for(white_level: int) -> int:
    """ceil(log2(white_level + 1)) clamped to [10, 16]."""
    w = int(white_level)
    bits = math.ceil(math.log2(w + 1)) if w > 0 else 0
    return max(10, min(16, bits))


def normalize_wb(mul: Sequence[float] | None) -> tuple[float, float, float, float] | None:
    """[R, G, B, G2] multipliers normalised so G = 1. G2 == 0 -> G2 := G.

    Returns None when the input is unusable (missing, non-finite, or R/G/B <= 0).
    """
    if mul is None:
        return None
    m = [float(x) for x in mul]
    if len(m) < 3 or not all(math.isfinite(x) for x in m):
        return None
    r, g, b = m[0], m[1], m[2]
    g2 = m[3] if len(m) > 3 else 0.0
    if r <= 0 or g <= 0 or b <= 0:
        return None
    if g2 <= 0:
        g2 = g
    return (r / g, 1.0, b / g, g2 / g)


def is_valid_matrix(m: Any, *, rows_sum_to_one: bool = False, tol: float = 0.02) -> bool:
    """3x3, finite, not all zero, invertible; optionally every row sums to 1 ± tol."""
    try:
        a = np.asarray(m, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if a.shape != (3, 3) or not np.all(np.isfinite(a)) or not np.any(a != 0):
        return False
    if abs(np.linalg.det(a)) < 1e-9:
        return False
    if rows_sum_to_one and not np.all(np.abs(a.sum(axis=1) - 1.0) <= tol):
        return False
    return True


def rgb_cam_from_cam_xyz(cam_xyz: Any) -> np.ndarray | None:
    """dcraw/LibRaw ``cam_xyz_coeff``: XYZ->camera matrix -> camera->sRGB(linear D65) matrix.

    cam_rgb = cam_xyz · xyz_rgb; each row normalised to sum 1 (so camera white
    maps to sRGB white); rgb_cam = inverse(cam_rgb).  Returns None if unusable.
    """
    a = np.asarray(cam_xyz, dtype=np.float64)
    if a.shape != (3, 3) or not is_valid_matrix(a):
        return None
    cam_rgb = a @ _XYZ_RGB
    sums = cam_rgb.sum(axis=1)
    if np.any(np.abs(sums) < 1e-9):
        return None
    cam_rgb = cam_rgb / sums[:, None]
    try:
        rgb_cam = np.linalg.inv(cam_rgb)
    except np.linalg.LinAlgError:
        return None
    return rgb_cam if is_valid_matrix(rgb_cam, rows_sum_to_one=True) else None


def even_geometry(
    raw_width: int, raw_height: int, left: int, top: int, vis_width: int, vis_height: int
) -> tuple[int, int, tuple[int, int, int, int]]:
    """(width, height, (crop_left, crop_top, crop_width, crop_height)) for HEAD.

    Odd mosaic dimensions are trimmed to even by dropping the LAST column/row
    (keeps the CFA phase at (0,0)).  The visible area is clipped to the trimmed
    mosaic; if it covers it entirely the crop is (0, 0, 0, 0) ("full image").
    """
    w = int(raw_width) & ~1
    h = int(raw_height) & ~1
    if w < 2 or h < 2:
        raise ValueError(f"mosaic too small: {raw_width}x{raw_height}")
    left, top = int(left), int(top)
    vw = min(int(vis_width), w - left)
    vh = min(int(vis_height), h - top)
    if left < 0 or top < 0 or vw <= 0 or vh <= 0:
        raise ValueError(f"visible area ({left},{top},{vis_width}x{vis_height}) outside mosaic {w}x{h}")
    crop = (0, 0, 0, 0) if (left, top, vw, vh) == (0, 0, w, h) else (left, top, vw, vh)
    return w, h, crop


def _f32(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(values, dtype=np.float32).reshape(-1))


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0:
        return None
    return float(f"{v:.6g}")


# ---------------------------------------------------------------------------
# Make / Model (rawpy 0.28 does not expose idata.make / idata.model)
# ---------------------------------------------------------------------------
_TAG_MAKE, _TAG_MODEL = 0x010F, 0x0110
_HEADER_SCAN = 1 << 20


def _tiff_ascii_tags(buf: bytes, base: int) -> dict[str, str]:
    """Make/Model from IFD0 of a TIFF structure starting at ``buf[base]``. Best effort."""
    out: dict[str, str] = {}
    try:
        bo = {b"II": "<", b"MM": ">"}.get(buf[base : base + 2])
        if bo is None:
            return out
        (ifd,) = struct.unpack_from(bo + "I", buf, base + 4)
        pos = base + ifd
        (n,) = struct.unpack_from(bo + "H", buf, pos)
        for i in range(min(n, 512)):
            tag, typ, count = struct.unpack_from(bo + "HHI", buf, pos + 2 + 12 * i)
            if tag not in (_TAG_MAKE, _TAG_MODEL) or typ != 2 or count == 0 or count > 256:
                continue
            if count <= 4:
                raw = buf[pos + 2 + 12 * i + 8 : pos + 2 + 12 * i + 8 + count]
            else:
                (off,) = struct.unpack_from(bo + "I", buf, pos + 2 + 12 * i + 8)
                raw = buf[base + off : base + off + count]
            text = raw.split(b"\x00", 1)[0].decode("utf-8", "replace").strip()
            if text:
                out["make" if tag == _TAG_MAKE else "model"] = text
    except (struct.error, IndexError):
        pass
    return out


def _read_make_model_from_file(path: str) -> dict[str, str]:
    """IFD0 Make/Model for TIFF-based RAWs (CR2/NEF/NRW/ARW/SR2/DNG/ORF/RW2/PEF) and CR3 (CMT1 box)."""
    try:
        with open(path, "rb") as f:
            buf = f.read(_HEADER_SCAN)
    except OSError:
        return {}
    if buf[:2] in (b"II", b"MM"):
        return _tiff_ascii_tags(buf, 0)
    i = buf.find(b"CMT1")  # Canon CR3 (ISO-BMFF): CMT1 box payload is a TIFF IFD0
    if i >= 4:
        return _tiff_ascii_tags(buf, i + 4)
    return {}


def _read_make_model_from_jpeg(data: bytes) -> dict[str, str]:
    try:
        image_mod = _import_pil()
        with image_mod.open(io.BytesIO(data)) as im:
            ex = im.getexif()
    except Exception:  # noqa: BLE001 - best effort, never fatal
        return {}
    out = {}
    for tag, key in ((_TAG_MAKE, "make"), (_TAG_MODEL, "model")):
        v = ex.get(tag)
        if isinstance(v, bytes):
            v = v.decode("utf-8", "replace")
        if isinstance(v, str) and v.strip("\x00 "):
            out[key] = v.strip("\x00 ")
    return out


# ---------------------------------------------------------------------------
# Builders from a rawpy-like object
# ---------------------------------------------------------------------------
def _check_bayer(raw: Any) -> np.ndarray:
    raw_type = getattr(raw, "raw_type", None)
    if raw_type is not None and getattr(raw_type, "name", str(raw_type)) not in ("Flat", "RawType.Flat"):
        raise ValueError("RAW is not a flat CFA mosaic (Foveon / linear / multi-shot); only 2x2 Bayer is supported")
    if int(raw.num_colors) != 3:
        raise ValueError(f"num_colors = {raw.num_colors}; only 3-colour RGB Bayer sensors are supported")
    try:
        pattern = raw.raw_pattern
    except NotImplementedError as e:
        raise ValueError(f"unsupported colour filter array: {e}") from None
    if pattern is None:
        raise ValueError("RAW has no CFA pattern (not a flat Bayer image)")
    pattern = np.asarray(pattern)
    if pattern.shape != (2, 2):
        raise ValueError(
            f"CFA pattern is {pattern.shape[0]}x{pattern.shape[1]} (e.g. Fujifilm X-Trans); only 2x2 Bayer is supported"
        )
    img = raw.raw_image
    if img.ndim != 2:
        raise ValueError("raw_image is not a 2-D mosaic; only 2x2 Bayer is supported")
    return pattern


def _full_pattern_indices(raw: Any) -> list[list[int]]:
    """LibRaw colour indices (0..3) of the 2x2 block at full-mosaic (0,0)."""
    s = raw.sizes
    top, left = int(s.top_margin), int(s.left_margin)
    # LibRaw COLOR()/filters are relative to the visible area; raw_color(row, col)
    # takes FULL-image coordinates (rawpy subtracts the margins). Read the
    # pattern at the visible origin and shift it by the margins.
    vis = [[int(raw.raw_color(top + y, left + x)) for x in range(2)] for y in range(2)]
    full = pattern_full_from_visible(vis, top, left)
    check = [[int(raw.raw_color(y, x)) for x in range(2)] for y in range(2)]
    if full != check:  # pragma: no cover - would indicate a LibRaw/rawpy inconsistency
        raise ValueError(f"inconsistent CFA phase: shifted {full} vs raw_color {check}")
    return full


def build_head(raw: Any) -> tuple[Head, np.ndarray]:
    """HEAD + uint16 mosaic (copied, trimmed to even) from a rawpy-like object."""
    _check_bayer(raw)
    img = np.asarray(raw.raw_image)
    if img.dtype != np.uint16:
        if img.dtype.kind != "u" or img.dtype.itemsize > 2:
            raise ValueError(f"unexpected raw_image dtype {img.dtype}")
        img = img.astype(np.uint16)
    s = raw.sizes
    h_full, w_full = img.shape
    width, height, crop = even_geometry(w_full, h_full, s.left_margin, s.top_margin, s.width, s.height)
    mosaic = np.array(img[:height, :width], dtype=np.uint16, copy=True, order="C")

    idx = _full_pattern_indices(raw)
    cfa = map_cfa_pattern(idx, raw.color_desc)
    black = black_levels_for_pattern(idx, raw.black_level_per_channel)
    white = int(raw.white_level)
    if white > 65535:
        raise ValueError(f"white_level {white} exceeds 16 bits")
    if not max(black) < white:
        raise ValueError(f"black level {max(black)} >= white level {white}")
    head = Head(
        width=width,
        height=height,
        effective_bits=effective_bits_for(white),
        white_level=white,
        cfa_pattern=cfa,
        black_level=black,
        orientation=flip_to_orientation(s.flip),
        crop_left=crop[0],
        crop_top=crop[1],
        crop_width=crop[2],
        crop_height=crop[3],
    )
    head.validate()
    return head, mosaic


def build_color(raw: Any) -> Color:
    as_shot = normalize_wb(getattr(raw, "camera_whitebalance", None))
    daylight = normalize_wb(getattr(raw, "daylight_whitebalance", None))
    ones = (1.0, 1.0, 1.0, 1.0)
    flags = 0

    cam_xyz = np.zeros((3, 3))
    try:
        cand = np.asarray(raw.rgb_xyz_matrix, dtype=np.float64)[:3, :3]
        if cand.shape == (3, 3) and np.all(np.isfinite(cand)) and np.any(cand != 0):
            cam_xyz = cand
            flags |= COLR_FLAG_CAM_XYZ
    except (AttributeError, TypeError, ValueError):
        pass

    rgb_cam = rgb_cam_from_cam_xyz(cam_xyz) if flags & COLR_FLAG_CAM_XYZ else None
    if rgb_cam is None:
        try:
            cm = np.asarray(raw.color_matrix, dtype=np.float64)[:3, :3]
            if is_valid_matrix(cm, rows_sum_to_one=True):
                rgb_cam = cm
        except (AttributeError, TypeError, ValueError):
            pass
    if rgb_cam is not None:
        flags |= COLR_FLAG_RGB_CAM
    else:
        rgb_cam = np.eye(3)

    # Values are rounded to f32 here so the in-memory Color equals what COLR stores.
    return Color(
        as_shot_wb=_f32(as_shot or daylight or ones),
        daylight_wb=_f32(daylight or ones),
        rgb_cam=_f32(rgb_cam.reshape(-1)),
        cam_xyz=_f32(cam_xyz.reshape(-1)),
        illuminant=ILLUMINANT_D65,
        flags=flags,
    )


def build_meta(
    raw: Any,
    src_path: str | os.PathLike | None,
    thumb_jpeg: bytes | None = None,
    *,
    make_model: dict[str, str] | None = None,
) -> dict:
    """META JSON. Make/Model come from the RAW's own TIFF IFD0 (or CR3 CMT1 box),
    falling back to the EXIF of the embedded JPEG preview; omitted if neither has them."""
    meta: dict[str, Any] = {}
    mm = dict(make_model) if make_model is not None else (
        _read_make_model_from_file(os.fspath(src_path)) if src_path is not None else {}
    )
    if len(mm) < 2 and thumb_jpeg:
        mm = {**_read_make_model_from_jpeg(thumb_jpeg), **mm}
    meta.update(mm)
    try:
        lens = raw.lens
        model = (getattr(lens, "model", "") or "").strip("\x00 ")
        if model:
            meta["lens"] = model
    except Exception:  # noqa: BLE001 - lens is optional metadata
        pass
    try:
        other = raw.other
    except Exception:  # noqa: BLE001 - e.g. timestamp out of platform range
        other = None
    if other is not None:
        iso = _num(getattr(other, "iso_speed", None))
        if iso is not None:
            meta["iso"] = int(iso) if float(iso).is_integer() else iso
        for key, attr in (("shutter_s", "shutter_speed"), ("aperture", "aperture"), ("focal_mm", "focal_length")):
            v = _num(getattr(other, attr, None))
            if v is not None:
                meta[key] = v
        ts = getattr(other, "timestamp", None)
        if isinstance(ts, _dt.datetime) and ts.year > 1970:
            meta["datetime"] = ts.replace(tzinfo=None, microsecond=0).isoformat(timespec="seconds")
        artist = (getattr(other, "artist", "") or "").strip("\x00 ")
        if artist:
            meta["artist"] = artist
    if src_path is not None:
        name = os.path.basename(os.fspath(src_path))
        meta["source_file"] = name
        ext = os.path.splitext(name)[1]
        if ext:
            meta["source_format"] = ext[1:].upper()
    meta["software"] = SOFTWARE
    return meta


# ---------------------------------------------------------------------------
# Thumbnail
# ---------------------------------------------------------------------------
def _extract_embedded(raw: Any) -> tuple[str, Any] | None:
    """('jpeg', bytes) | ('bitmap', HxWx3 uint8) | None if the file has no usable preview."""
    errors = _rawpy_thumb_errors()
    try:
        t = raw.extract_thumb()
    except errors:
        return None
    kind = getattr(t.format, "name", str(t.format)).upper()
    if "JPEG" in kind:
        return ("jpeg", bytes(t.data))
    if "BITMAP" in kind:
        arr = np.asarray(t.data)
        if arr.ndim == 3 and arr.shape[2] >= 3 and arr.dtype == np.uint8:
            return ("bitmap", np.ascontiguousarray(arr[:, :, :3]))
    return None


def _encode_jpeg(im: Any, thumb_max: int | None) -> Thumbnail:
    image_mod = _import_pil()
    im = im.convert("RGB")
    if thumb_max is not None and max(im.size) > thumb_max:
        im.thumbnail((thumb_max, thumb_max), image_mod.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=THUMB_JPEG_QUALITY)
    return Thumbnail(format=0, width=im.size[0], height=im.size[1], data=buf.getvalue())


def build_thumbnail(raw: Any, embedded: tuple[str, Any] | None, thumb_max: int | None = 512) -> Thumbnail:
    """THMB from the embedded preview, or from a half-size LibRaw render if there is none.

    JPEG previews within ``thumb_max`` are stored verbatim (format 0); larger ones
    are resized and re-encoded as JPEG q85.  Bitmap previews within the limit are
    stored as RGB8 (format 1).  The generated fallback uses ``user_flip=0`` so it
    is unrotated, like embedded previews and the mosaic (HEAD.orientation applies).
    """
    image_mod = _import_pil()
    if embedded is not None and embedded[0] == "jpeg":
        try:
            with image_mod.open(io.BytesIO(embedded[1])) as im:
                im.load()
                w, h = im.size
                if thumb_max is None or max(w, h) <= thumb_max:
                    return Thumbnail(format=0, width=w, height=h, data=embedded[1])
                return _encode_jpeg(im, thumb_max)
        except (OSError, ValueError):
            embedded = None  # undecodable preview -> regenerate below
    if embedded is not None and embedded[0] == "bitmap":
        arr = embedded[1]
        h, w = arr.shape[:2]
        if thumb_max is None or max(w, h) <= thumb_max:
            return Thumbnail(format=1, width=w, height=h, data=arr.tobytes())
        return _encode_jpeg(image_mod.fromarray(arr), thumb_max)
    rgb = raw.postprocess(half_size=True, use_camera_wb=True, user_flip=0, output_bps=8)
    return _encode_jpeg(image_mod.fromarray(np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)), thumb_max)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def from_rawpy(
    raw: Any,
    src_path: str | os.PathLike | None = None,
    *,
    thumbnail: bool = True,
    thumb_max: int | None = 512,
) -> FrawFile:
    """Build a FrawFile from an open rawpy.RawPy (or any object with the same interface)."""
    if thumb_max is not None and thumb_max < 1:
        raise ValueError("thumb_max must be >= 1 or None")
    head, mosaic = build_head(raw)
    color = build_color(raw)
    file_mm = _read_make_model_from_file(os.fspath(src_path)) if src_path is not None else {}
    # The embedded preview is needed for THMB, and as an EXIF fallback for Make/Model.
    need_exif = src_path is not None and len(file_mm) < 2
    embedded = _extract_embedded(raw) if (thumbnail or need_exif) else None
    thumb = build_thumbnail(raw, embedded, thumb_max) if thumbnail else None
    meta = build_meta(raw, src_path, embedded[1] if embedded and embedded[0] == "jpeg" else None, make_model=file_mm)
    f = FrawFile(head=head, raw=mosaic, meta=meta, color=color, thumbnail=thumb)
    f.validate()
    return f


def _check_source(src_path: str | os.PathLike) -> str:
    path = os.fspath(src_path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        exts = " ".join(sorted(e[1:].upper() for e in SUPPORTED_EXTENSIONS))
        raise ValueError(f"unsupported RAW extension {ext or '(none)'!r}; supported: {exts}")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def ingest(src_path: str | os.PathLike, *, thumbnail: bool = True, thumb_max: int | None = 512) -> FrawFile:
    """Read a camera RAW file and return the equivalent FrawFile (not written to disk).

    Raises ValueError for unsupported extensions and non-Bayer sensors (X-Trans,
    Foveon, 4-colour, ...), ImportError if rawpy is missing; LibRaw read errors
    (``rawpy.LibRawError`` subclasses) propagate unchanged.
    """
    path = _check_source(src_path)
    rawpy = _import_rawpy()
    with rawpy.imread(path) as raw:
        return from_rawpy(raw, path, thumbnail=thumbnail, thumb_max=thumb_max)


def ingest_to_file(
    src: str | os.PathLike,
    dst: str | os.PathLike,
    compress: bool = True,
    *,
    thumbnail: bool = True,
    thumb_max: int | None = 512,
) -> FrawFile:
    """ingest() + write to ``dst`` (.fraw/.frw). Returns the FrawFile."""
    f = ingest(src, thumbnail=thumbnail, thumb_max=thumb_max)
    f.write(dst, compress=compress)
    return f


def ingest_burst(
    srcs: Iterable[str | os.PathLike],
    dst_fset: str | os.PathLike,
    burst_id: str | None = None,
    *,
    compress: bool = True,
    thumbnail: bool = True,
    thumb_max: int | None = 512,
) -> FrawSet:
    """Ingest several RAWs into one .fset burst container. Returns the FrawSet written."""
    paths = list(srcs)
    if not paths:
        raise ValueError("ingest_burst needs at least one source file")
    frames = [ingest(p, thumbnail=thumbnail, thumb_max=thumb_max) for p in paths]
    fset = FrawSet(
        frames=frames,
        burst_id=burst_id if burst_id is not None else uuid.uuid4().hex,
        created=_dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    )
    fset.write(dst_fset, compress=compress)
    return fset
