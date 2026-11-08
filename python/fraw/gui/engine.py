"""Adaptador Qt-free entre la interfaz y el motor de revelado (:mod:`fraw.processor`).

La interfaz nunca llama al procesador directamente: pasa por :class:`Engine`, que
detecta qué partes de la API del motor existen (``preview(draft=, apply_crop=)``,
``render_region``, ``auto_tone``, ``auto_wb``, ``as_shot_kelvin``, ``sample_wb``,
``histogram``, ``clipping_mask``, ``export(long_edge=, output_sharpen=, …)``) y,
para lo que falte, usa un *fallback* en numpy para que la interfaz siga
funcionando. ``Engine.missing()`` lista lo que se está emulando.

Coordenadas: ``rect_norm`` / ``(xn, yn)`` están normalizadas (0…1) sobre la
imagen que muestra el lienzo (con recorte aplicado salvo que se pida lo contrario).
"""

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .. import processor as _pmod
from ..devparams import SPECS, DevParams
from .render import neutral_wb_from_patch, oriented_size, patch_cam_rgb

__all__ = [
    "Engine",
    "histogram_np",
    "clipping_mask_np",
    "apply_geometry_np",
    "geometry_size",
    "effective_params",
    "before_params",
    "GEOMETRY_KEYS",
]

GEOMETRY_KEYS = tuple(s.key for s in SPECS.values() if s.group == "geometry")
_REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


# --------------------------------------------------------------------- helpers
def effective_params(params: DevParams, bypass: set[str] | frozenset[str] = frozenset()) -> DevParams:
    """Copia de ``params`` con los grupos en ``bypass`` en su valor por defecto."""
    p = DevParams.coerce(params)
    for g in bypass:
        p.reset_group(g)
    return p


def before_params(params: DevParams) -> DevParams:
    """Parámetros "Antes": todo por defecto salvo la geometría (para que alinee)."""
    p = DevParams()
    cur = DevParams.coerce(params)
    for k in GEOMETRY_KEYS:
        setattr(p, k, getattr(cur, k))
    p.demosaic = cur.demosaic
    return p


def histogram_np(img: np.ndarray, max_samples: int = 120_000) -> dict[str, Any]:
    """Histograma RGB + luminancia (256 bins) y fracción de píxeles recortados (submuestreo en grilla)."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    k = max(1, int((h * w / max_samples) ** 0.5))
    s = a[::k, ::k].reshape(-1, 3)
    out: dict[str, Any] = {}
    r, g, b = s[:, 0], s[:, 1], s[:, 2]
    for c, ch in zip("rgb", (r, g, b)):
        out[c] = np.bincount(ch, minlength=256)[:256].astype(np.float32)
    lum = ((54 * r.astype(np.uint16) + 183 * g.astype(np.uint16) + 19 * b.astype(np.uint16)) >> 8).astype(np.uint8)
    out["l"] = np.bincount(lum, minlength=256)[:256].astype(np.float32)
    n = max(1, s.shape[0])
    out["clip_shadows"] = float(np.count_nonzero((r == 0) | (g == 0) | (b == 0))) / n
    out["clip_highlights"] = float(np.count_nonzero((r == 255) | (g == 255) | (b == 255))) / n
    return out


def clipping_mask_np(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sombras, luces) booleanas H×W: algún canal en 0 / algún canal en 255."""
    a = np.asarray(img)
    return a.min(axis=2) == 0, a.max(axis=2) == 255


def geometry_size(base_wh: tuple[int, int], params: DevParams | Mapping[str, Any], apply_crop: bool = True) -> tuple[int, int]:
    """Tamaño (w, h) de la imagen revelada a resolución completa tras giro 90° y recorte."""
    p = DevParams.coerce(params)
    w, h = base_wh
    if p.rotate_90 in ("90", "270"):
        w, h = h, w
    if apply_crop:
        cw = max(1e-3, p.crop_right - p.crop_left)
        ch = max(1e-3, p.crop_bottom - p.crop_top)
        w, h = max(1, int(round(w * cw))), max(1, int(round(h * ch)))
    return int(w), int(h)


def apply_geometry_np(img: np.ndarray, params: DevParams | Mapping[str, Any], apply_crop: bool = True) -> np.ndarray:
    """Fallback de geometría (girar 90°, voltear, recorte) sobre una imagen ya revelada."""
    p = DevParams.coerce(params)
    out = img
    k = {"0": 0, "90": -1, "180": 2, "270": 1}[p.rotate_90]
    if k:
        out = np.rot90(out, k)
    if p.flip_h:
        out = out[:, ::-1]
    if p.flip_v:
        out = out[::-1, :]
    if apply_crop and (p.crop_left > 0 or p.crop_top > 0 or p.crop_right < 1 or p.crop_bottom < 1):
        h, w = out.shape[:2]
        x0, x1 = int(round(p.crop_left * w)), int(round(p.crop_right * w))
        y0, y1 = int(round(p.crop_top * h)), int(round(p.crop_bottom * h))
        x1, y1 = max(x0 + 1, x1), max(y0 + 1, y1)
        out = out[y0:y1, x0:x1]
    return np.ascontiguousarray(out)


def _accepts(fn: Any, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _srgb_decode(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


# --------------------------------------------------------------------- Engine
class Engine:
    """Fachada sobre un :class:`~fraw.processor.FRAWProcessor` con fallbacks."""

    def __init__(self, proc: Any):
        self.proc = proc
        pv = getattr(proc, "preview", None)
        self._pv_draft = pv is not None and _accepts(pv, "draft")
        self._pv_crop = pv is not None and _accepts(pv, "apply_crop")
        ex = getattr(proc, "export", None)
        self._export_kw = {
            k for k in ("bits", "quality", "long_edge", "output_sharpen", "embed_icc", "metadata", "demosaic")
            if ex is not None and _accepts(ex, k)
        }

    # ------------------------------------------------------------ capability
    def has(self, name: str) -> bool:
        if name == "preview_draft":
            return self._pv_draft
        if name == "preview_crop":
            return self._pv_crop
        if name in ("histogram", "clipping_mask", "kelvin_to_multipliers", "multipliers_to_kelvin"):
            return callable(getattr(self.proc, name, None)) or callable(getattr(_pmod, name, None))
        return callable(getattr(self.proc, name, None))

    def missing(self) -> list[str]:
        names = ["preview_draft", "preview_crop", "render_region", "auto_tone", "auto_wb", "as_shot_kelvin",
                 "sample_wb", "pixel_values", "histogram", "clipping_mask"]
        out = [n for n in names if not self.has(n)]
        for k in ("long_edge", "output_sharpen", "embed_icc", "metadata"):
            if k not in self._export_kw:
                out.append(f"export({k}=)")
        return out

    # --------------------------------------------------------------- sizes
    def base_size(self) -> tuple[int, int]:
        """(w, h) orientado, sin geometría de revelado."""
        fn = getattr(self.proc, "oriented_size", None)
        if callable(fn):
            try:
                w, h = fn()
                return int(w), int(h)
            except Exception:  # noqa: BLE001
                pass
        norm = getattr(self.proc, "norm", None)
        if norm is not None:
            h, w = norm.shape[:2]
            return oriented_size(w, h, getattr(self.proc, "orientation", 1))
        head = self.proc.head
        return oriented_size(head.width, head.height, head.orientation)

    def output_size(self, params: DevParams, apply_crop: bool = True) -> tuple[int, int]:
        fn = getattr(self.proc, "output_size", None)
        if callable(fn):
            try:
                w, h = fn(params, apply_crop=apply_crop)
                return int(w), int(h)
            except Exception:  # noqa: BLE001
                pass
        return geometry_size(self.base_size(), params, apply_crop)

    # ------------------------------------------------------------- preview
    def preview(self, params: DevParams, max_side: int = 1600, *, apply_crop: bool = True, draft: bool = False) -> np.ndarray:
        kw: dict[str, Any] = {}
        if self._pv_crop:
            kw["apply_crop"] = apply_crop
        if self._pv_draft:
            kw["draft"] = draft
        img = self.proc.preview(params, max_side, **kw)
        if not self._pv_crop:
            img = apply_geometry_np(img, params, apply_crop)
        return np.ascontiguousarray(img)

    def render_region(self, params: DevParams, rect_norm: tuple[float, float, float, float],
                      out_w: int, out_h: int) -> np.ndarray | None:
        fn = getattr(self.proc, "render_region", None)
        if callable(fn):
            return np.ascontiguousarray(fn(params, tuple(float(v) for v in rect_norm), int(out_w), int(out_h)))
        return None

    # ------------------------------------------------------------ analysis
    def histogram(self, img: np.ndarray) -> dict[str, Any]:
        fn = getattr(self.proc, "histogram", None) or getattr(_pmod, "histogram", None)
        if callable(fn):
            try:
                a = np.asarray(img)
                k = max(1, int((a.shape[0] * a.shape[1] / 250_000) ** 0.5))
                h = fn(np.ascontiguousarray(a[::k, ::k]) if k > 1 else a)  # submuestreo: el histograma no necesita todo
                if isinstance(h, Mapping) and all(k in h for k in ("r", "g", "b")):
                    out = dict(h)
                    if "l" not in out:
                        out["l"] = histogram_np(img)["l"]
                    for k in "rgbl":
                        out[k] = np.asarray(out[k], dtype=np.float32)[:256]
                    out.setdefault("clip_shadows", 0.0)
                    out.setdefault("clip_highlights", 0.0)
                    return out
            except Exception:  # noqa: BLE001
                pass
        return histogram_np(img)

    def clipping_mask(self, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fn = getattr(self.proc, "clipping_mask", None) or getattr(_pmod, "clipping_mask", None)
        if callable(fn):
            try:
                sh, hl = fn(img)
                sh, hl = np.asarray(sh, dtype=bool), np.asarray(hl, dtype=bool)
                if sh.shape == img.shape[:2] and hl.shape == img.shape[:2]:
                    return sh, hl
            except Exception:  # noqa: BLE001
                pass
        return clipping_mask_np(img)

    # ---------------------------------------------------------------- WB
    def as_shot_kelvin(self) -> tuple[float, float] | None:
        fn = getattr(self.proc, "as_shot_kelvin", None)
        if callable(fn):
            try:
                k, t = fn()
                return float(k), float(t)
            except Exception:  # noqa: BLE001
                return None
        m2k = getattr(_pmod, "multipliers_to_kelvin", None)
        color = getattr(self.proc, "color", None)
        if callable(m2k) and color is not None:
            try:
                k, t = m2k(color, color.as_shot_wb)  # type: ignore[call-arg]
                return float(k), float(t)
            except Exception:  # noqa: BLE001
                return None
        return None

    def auto_wb(self) -> dict[str, Any]:
        fn = getattr(self.proc, "auto_wb", None)
        if callable(fn):
            r = fn()
            if isinstance(r, Mapping):
                return dict(r)
        return self._auto_wb_fallback()

    def _auto_wb_fallback(self) -> dict[str, Any]:
        norm = getattr(self.proc, "norm", None)
        pattern = getattr(self.proc, "pattern", None)
        if norm is None or pattern is None:
            return {}
        step = max(2, (int(max(norm.shape)) // 600) & ~1)
        acc = np.zeros(3)
        for i, c in enumerate(pattern):
            v = norm[i >> 1 :: step, i & 1 :: step]
            ok = v[(v < 0.95) & (v > 0.002)]
            if ok.size:
                acc[c] += float(ok.mean()) / max(1, list(pattern).count(c))
        wb = neutral_wb_from_patch(acc)
        if wb is None:
            return {}
        return {"wb_mode": "custom", "wb_custom": list(wb), "temperature_shift": 0.0, "tint_shift": 0.0}

    def sample_wb(self, xn: float, yn: float, radius: int, params: DevParams,
                  img_size: tuple[int, int] | None = None) -> dict[str, Any] | None:
        """Cuentagotas en coordenadas normalizadas de la imagen mostrada."""
        fn = getattr(self.proc, "sample_wb", None)
        if callable(fn):
            try:
                r = fn(xn, yn, radius, params=params)
            except TypeError:
                r = fn(xn, yn, radius)
            if isinstance(r, Mapping):
                return dict(r)
            if isinstance(r, (tuple, list)) and len(r) == 3:
                return {"wb_mode": "custom", "wb_custom": [float(v) for v in r]}
            return None
        # fallback: sólo exacto sin geometría (la mayoría de los casos de cuentagotas)
        if getattr(self.proc, "norm", None) is None:
            return None
        w, h = self.base_size()
        cam = patch_cam_rgb(self.proc, xn * w, yn * h, max(1, int(radius)))
        wb = neutral_wb_from_patch(cam)
        if wb is None:
            return None
        return {"wb_mode": "custom", "wb_custom": list(wb), "temperature_shift": 0.0, "tint_shift": 0.0}

    # --------------------------------------------------------------- tone
    def auto_tone(self, params: DevParams) -> dict[str, Any]:
        fn = getattr(self.proc, "auto_tone", None)
        if callable(fn):
            r = fn(params)
            if isinstance(r, Mapping):
                return dict(r)
        return self._auto_tone_fallback(params)

    def _auto_tone_fallback(self, params: DevParams) -> dict[str, Any]:
        p = DevParams.coerce(params)
        for k in ("exposure_ev", "contrast", "highlights", "shadows", "whites", "blacks"):
            p.reset([k])
        img = self.preview(p, 512, apply_crop=True)
        y = _srgb_decode(img.astype(np.float32) / 255.0) @ _REC709
        y = y.ravel()
        med = float(np.median(y))
        p995 = float(np.percentile(y, 99.5))
        p005 = float(np.percentile(y, 0.5))
        ev = 0.0 if med <= 1e-4 else float(np.clip(math.log2(0.18 / med), -4.0, 4.0))
        ev = min(ev, math.log2(1.0 / max(p995, 1e-4)) + 1.0)  # no quemar más de 1 EV de altas luces
        hl = float(np.clip(-(p995 * 2**ev - 0.9) * 0.8, -1.0, 0.0))
        sh = float(np.clip((0.05 - med * 2**ev) * 4.0, 0.0, 0.6))
        bl = float(np.clip(-(p005 * 2**ev) * 8.0, -0.4, 0.0))
        return {"exposure_ev": round(ev, 2), "contrast": 0.1, "highlights": round(hl, 2),
                "shadows": round(sh, 2), "whites": 0.0, "blacks": round(bl, 2)}

    # ------------------------------------------------------------- export
    def export(self, path: str | os.PathLike, params: DevParams, *, bits: int = 8, quality: int = 95,
               long_edge: int | None = None, output_sharpen: str = "none", embed_icc: bool = True,
               metadata: bool = True, demosaic: str | None = None, meta: Mapping[str, Any] | None = None) -> Path:
        path = Path(path)
        opts = {"bits": bits, "quality": quality, "long_edge": long_edge, "output_sharpen": output_sharpen,
                "embed_icc": embed_icc, "metadata": metadata, "demosaic": demosaic}
        need = {k for k, v in opts.items() if k in ("long_edge", "output_sharpen", "embed_icc", "metadata")
                and v not in (None, False, "none", 0)}
        if need <= self._export_kw and self._pv_crop:
            kw = {k: v for k, v in opts.items() if k in self._export_kw}
            r = self.proc.export(path, params, **kw)
            return Path(r) if r else path
        return self._export_fallback(path, params, meta=meta, **opts)

    def _export_fallback(self, path: Path, params: DevParams, *, bits: int, quality: int, long_edge: int | None,
                         output_sharpen: str, embed_icc: bool, metadata: bool, demosaic: str | None,
                         meta: Mapping[str, Any] | None) -> Path:
        from PIL import Image  # noqa: PLC0415

        ext = path.suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
            raise ValueError(f"formato de exportación no soportado: {ext}")
        if ext in (".jpg", ".jpeg"):
            bits = 8
        img = self.proc.develop(params, demosaic=demosaic) if _accepts(self.proc.develop, "demosaic") else self.proc.develop(params)
        if not self._pv_crop:
            img = apply_geometry_np(img, params, True)
        img = np.asarray(img, dtype=np.float32)
        h, w = img.shape[:2]
        if long_edge and max(w, h) > long_edge:
            f = long_edge / max(w, h)
            nw, nh = max(1, round(w * f)), max(1, round(h * f))
            img = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(img[..., c])).resize((nw, nh), Image.Resampling.LANCZOS))
                            for c in range(3)], axis=2)
        amount = {"none": 0.0, "screen": 0.35, "matte": 0.6, "glossy": 0.5}.get(str(output_sharpen), 0.0)
        if amount > 0:
            from scipy.ndimage import gaussian_filter  # noqa: PLC0415

            blur = np.stack([gaussian_filter(img[..., c], 0.8) for c in range(3)], axis=2)
            img = img + amount * (img - blur)
        img = np.clip(img, 0.0, 1.0)
        if bits == 16:
            a16 = _pmod.to_uint16(img)
            if ext == ".png":
                _pmod.write_png16(path, a16)
            else:
                _pmod.write_tiff16(path, a16)
            return path
        im = Image.fromarray(np.ascontiguousarray(_pmod.to_uint8(img)))
        kw: dict[str, Any] = {}
        if embed_icc:
            try:
                from PIL import ImageCms  # noqa: PLC0415

                kw["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
            except Exception:  # noqa: BLE001
                pass
        if metadata and meta:
            ex = Image.Exif()
            for tag, key in ((0x010F, "make"), (0x0110, "model"), (0x0131, None)):
                if key and meta.get(key):
                    ex[tag] = str(meta[key])
            ex[0x0131] = "CTD Photo Studio"
            if meta.get("datetime"):
                ex[0x0132] = str(meta["datetime"]).replace("-", ":").replace("T", " ")[:19]
            kw["exif"] = ex.tobytes()
        if ext in (".jpg", ".jpeg"):
            im.save(path, "JPEG", quality=int(quality), subsampling=0, **kw)
        elif ext == ".png":
            kw.pop("exif", None) if not metadata else None
            im.save(path, "PNG", **kw)
        else:
            im.save(path, "TIFF", **kw)
        return path
