"""Output colour profile (sRGB ICC) and EXIF metadata for exports."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

__all__ = ["srgb_icc", "build_exif", "SOFTWARE"]

SOFTWARE = "CTD Photo Studio"


@lru_cache(maxsize=1)
def srgb_icc() -> bytes:
    """sRGB IEC61966-2.1 ICC profile bytes (LittleCMS built-in, via Pillow)."""
    try:
        from PIL import ImageCms

        return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    except Exception:  # noqa: BLE001 — Pillow without LittleCMS
        return b""


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _exif_datetime(s: Any) -> str | None:
    if not isinstance(s, str) or len(s) < 19:
        return None
    t = s[:19].replace("T", " ")
    if t[4] == "-" and t[7] == "-":
        t = t[:4] + ":" + t[5:7] + ":" + t[8:]
    return t


def build_exif(meta: Mapping[str, Any] | None, *, width: int | None = None, height: int | None = None):
    """Pillow ``Image.Exif`` with the basic tags (orientation 1: pixels are already upright)."""
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    m = dict(meta or {})
    ex = Image.Exif()
    for tag, key in ((271, "make"), (272, "model"), (315, "artist"), (33432, "copyright")):
        v = m.get(key)
        if isinstance(v, str) and v:
            ex[tag] = v
    ex[305] = SOFTWARE
    ex[274] = 1
    dt = _exif_datetime(m.get("datetime"))
    if dt:
        ex[306] = dt
    ifd = ex.get_ifd(0x8769)
    if dt:
        ifd[36867] = dt
        ifd[36868] = dt
    sh = _num(m.get("shutter_s"))
    if sh and sh > 0:
        ifd[33434] = IFDRational(1, int(round(1 / sh))) if sh < 1 else IFDRational(int(round(sh * 10)), 10)
    ap = _num(m.get("aperture"))
    if ap and ap > 0:
        ifd[33437] = IFDRational(int(round(ap * 10)), 10)
    iso = _num(m.get("iso"))
    if iso and iso > 0:
        ifd[34855] = int(round(iso))
    fl = _num(m.get("focal_mm"))
    if fl and fl > 0:
        ifd[37386] = IFDRational(int(round(fl * 10)), 10)
    lens = m.get("lens")
    if isinstance(lens, str) and lens:
        ifd[42036] = lens
    if width and height:
        ifd[40962] = int(width)
        ifd[40963] = int(height)
    ifd[40961] = 1  # ColorSpace sRGB
    return ex
