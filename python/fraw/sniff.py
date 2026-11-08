"""Detección del tipo real de un archivo por su CONTENIDO (no por la extensión).

Capas de decisión que usa CTD Photo Studio:

1. ``sniff()`` identifica el contenedor por sus bytes iniciales (magic bytes,
   cajas ISO-BMFF, cabecera TIFF y sus IFD).
2. Los formatos que ya son imágenes procesadas (JPEG, PNG, HEIC/AVIF, WebP,
   GIF, BMP, PSD, TIFF sin datos de sensor…) se rechazan con un motivo claro.
3. Todo lo demás (RAW conocidos o desconocidos) se entrega a LibRaw, y
   ``fraw.ingest`` sólo lo acepta si contiene el mosaico Bayer 2×2 original
   sin interpolar. Así un DNG renombrado a ``.tiff`` entra, y un DNG "rgb"
   (lineal / ya interpolado) o un Apple ProRAW se rechaza.

Este módulo es puro Python (sin numpy, rawpy ni Qt).
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

__all__ = ["Detected", "NotRawError", "sniff", "sniff_bytes", "HEAD_BYTES"]

#: Bytes iniciales que se leen para identificar el archivo.
HEAD_BYTES = 64 * 1024

_FRAW = b"FRAW\r\n\x1a\n"
_FSET = b"FSET\r\n\x1a\n"

# Etiquetas TIFF relevantes
_T_NEWSUBFILETYPE = 254
_T_MAKE = 271
_T_PHOTOMETRIC = 262
_T_SUBIFDS = 330
_T_EXIFIFD = 34665
_T_DNGVERSION = 50706
_PHOTO_CFA = 32803
_PHOTO_LINEARRAW = 34892

# ISO-BMFF: marcas "ftyp" de imágenes ya procesadas
_PROCESSED_BRANDS = {
    b"heic": "HEIC", b"heix": "HEIC", b"heim": "HEIC", b"heis": "HEIC",
    b"hevc": "HEIC", b"hevx": "HEIC", b"mif1": "HEIF", b"msf1": "HEIF",
    b"avif": "AVIF", b"avis": "AVIF",
}


class NotRawError(ValueError):
    """El archivo no es un RAW con datos de sensor (imagen procesada o formato desconocido)."""


@dataclass(frozen=True)
class Detected:
    """Resultado de :func:`sniff`.

    ``kind`` es uno de:

    * ``"fraw"`` / ``"fset"`` — archivo FabriRAW.
    * ``"raw"`` — RAW de cámara reconocido (se confirma luego con LibRaw).
    * ``"maybe_raw"`` — contenedor desconocido; se intenta con LibRaw.
    * ``"processed"`` — imagen ya procesada; se rechaza.
    * ``"empty"`` — archivo vacío o ilegible.
    """

    kind: str
    label: str
    reason: str | None = None

    @property
    def is_fabriraw(self) -> bool:
        return self.kind in ("fraw", "fset")

    @property
    def may_be_raw(self) -> bool:
        return self.kind in ("raw", "maybe_raw")


def _processed(label: str, why: str = "") -> Detected:
    reason = (
        f"es una imagen {label} ya procesada por la cámara o por un programa"
        + (f" ({why})" if why else "")
        + ": no contiene los datos originales del sensor. Usá el archivo RAW original."
    )
    return Detected("processed", label, reason)


# --------------------------------------------------------------------- TIFF
class _Tiff:
    """Lector mínimo y defensivo de IFDs TIFF sobre un buffer en memoria."""

    def __init__(self, buf: bytes, little: bool, base: int = 0):
        self.buf = buf
        self.e = "<" if little else ">"
        self.base = base

    def u16(self, off: int) -> int | None:
        off += self.base
        if off < 0 or off + 2 > len(self.buf):
            return None
        return struct.unpack_from(self.e + "H", self.buf, off)[0]

    def u32(self, off: int) -> int | None:
        off += self.base
        if off < 0 or off + 4 > len(self.buf):
            return None
        return struct.unpack_from(self.e + "I", self.buf, off)[0]

    def ifd(self, off: int) -> tuple[dict[int, tuple[int, int, int]], int | None]:
        """{tag: (type, count, value_or_offset)}, next_ifd_offset."""
        n = self.u16(off)
        if n is None or n == 0 or n > 1000:
            return {}, None
        tags: dict[int, tuple[int, int, int]] = {}
        for i in range(n):
            e = off + 2 + 12 * i
            tag, typ = self.u16(e), self.u16(e + 2)
            cnt, val = self.u32(e + 4), self.u32(e + 8)
            if None in (tag, typ, cnt, val):
                break
            if typ == 3 and cnt == 1:  # SHORT inline
                val = self.u16(e + 8) or 0
            tags[tag] = (typ, cnt, val)  # type: ignore[assignment]
        nxt = self.u32(off + 2 + 12 * n)
        return tags, nxt

    def longs(self, entry: tuple[int, int, int]) -> list[int]:
        typ, cnt, val = entry
        if cnt == 1 or typ not in (4, 13):
            return [val]
        out = []
        for i in range(min(cnt, 64)):
            v = self.u32(val + 4 * i)
            if v is None:
                break
            out.append(v)
        return out

    def ascii(self, entry: tuple[int, int, int]) -> str:
        typ, cnt, val = entry
        if typ != 2:
            return ""
        if cnt <= 4:
            raw = struct.pack(self.e + "I", val)[:cnt]
        else:
            o = val + self.base
            raw = self.buf[o : o + min(cnt, 128)]
        return raw.split(b"\0", 1)[0].decode("latin-1", "replace").strip()


def _tiff_survey(t: _Tiff, first: int) -> dict:
    """Recorre IFD0, la cadena, SubIFDs y EXIF (acotado) buscando señales de RAW."""
    info = {"dng": False, "cfa": False, "linear": False, "make": "", "photometrics": set(), "ifds": 0,
            "ifd0_main_rgb": False}
    seen: set[int] = set()
    queue = [first]
    while queue and info["ifds"] < 32:
        off = queue.pop(0)
        if not off or off in seen:
            continue
        seen.add(off)
        tags, nxt = t.ifd(off)
        if not tags:
            continue
        info["ifds"] += 1
        if off == first:
            sub = tags.get(_T_NEWSUBFILETYPE, (4, 1, 0))[2]
            ph0 = tags.get(_T_PHOTOMETRIC, (3, 1, -1))[2]
            info["ifd0_main_rgb"] = sub == 0 and ph0 in (1, 2)
        if _T_DNGVERSION in tags:
            info["dng"] = True
        if _T_MAKE in tags and not info["make"]:
            info["make"] = t.ascii(tags[_T_MAKE])
        if _T_PHOTOMETRIC in tags:
            ph = tags[_T_PHOTOMETRIC][2]
            info["photometrics"].add(ph)
            if ph == _PHOTO_CFA:
                info["cfa"] = True
            elif ph == _PHOTO_LINEARRAW:
                info["linear"] = True
        if _T_SUBIFDS in tags:
            queue.extend(t.longs(tags[_T_SUBIFDS]))
        if _T_EXIFIFD in tags:
            queue.append(tags[_T_EXIFIFD][2])
        if nxt:
            queue.append(nxt)
    return info


def _sniff_tiff(head: bytes) -> Detected:
    little = head[:2] == b"II"
    t = _Tiff(head, little)
    magic = t.u16(2)

    # Variantes TIFF de fabricantes
    if head[:4] in (b"IIRO", b"IIRS", b"MMOR"):
        return Detected("raw", "Olympus ORF")
    if magic == 0x55:
        return Detected("raw", "Panasonic RW2")
    if magic == 0x2B:  # BigTIFF: no lo usan las cámaras
        return _processed("BigTIFF")
    if magic != 42:
        return Detected("maybe_raw", "TIFF desconocido")
    if head[8:10] == b"CR":
        return Detected("raw", "Canon CR2")

    first = t.u32(4) or 0
    info = _tiff_survey(t, first)
    make = info["make"].upper()

    if info["dng"]:
        if info["linear"] and not info["cfa"]:
            return Detected(
                "processed", "DNG lineal",
                "es un DNG lineal (\"rgb\" / ProRAW): los colores ya fueron interpolados y "
                "no conserva el mosaico original del sensor.",
            )
        return Detected("raw", "DNG")
    if info["cfa"]:
        return Detected("raw", f"RAW TIFF{(' ' + info['make']) if make else ''}")
    # La imagen principal (IFD0, resolución completa) ya es RGB/gris: TIFF exportado
    # por un programa (Lightroom, escáner…), aunque conserve la marca de la cámara.
    if info["ifd0_main_rgb"] and not make.startswith(("NIKON", "SONY", "SAMSUNG")):
        return _processed("TIFF", "la imagen principal ya está en RGB")
    # TIFF de cámara sin CFA visible en los primeros KB (NEF/ARW/PEF/SRW…):
    # los datos pueden estar más adelante; que decida LibRaw.
    if make and info["ifds"] >= 2:
        return Detected("maybe_raw", f"RAW basado en TIFF ({info['make']})")
    if info["ifds"] >= 1 and info["photometrics"] and info["photometrics"] <= {0, 1, 2, 5, 6}:
        return _processed("TIFF", "RGB/escala de grises sin mosaico de sensor")
    return Detected("maybe_raw", "TIFF")


# --------------------------------------------------------------------- main
def sniff_bytes(head: bytes) -> Detected:
    """Identifica el formato a partir de los primeros bytes del archivo."""
    if not head:
        return Detected("empty", "vacío", "el archivo está vacío.")
    if head.startswith(_FRAW):
        return Detected("fraw", "FabriRAW")
    if head.startswith(_FSET):
        return Detected("fset", "FabriRAW ráfaga")

    # ISO-BMFF (CR3, HEIC, AVIF, MP4…)
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand == b"crx ":
            return Detected("raw", "Canon CR3")
        if brand in _PROCESSED_BRANDS:
            return _processed(_PROCESSED_BRANDS[brand])
        compat = head[16 : 16 + 4 * 8]
        for i in range(0, len(compat) - 3, 4):
            if compat[i : i + 4] == b"crx ":
                return Detected("raw", "Canon CR3")
            if compat[i : i + 4] in _PROCESSED_BRANDS:
                return _processed(_PROCESSED_BRANDS[compat[i : i + 4]])
        return Detected("maybe_raw", "contenedor ISO-BMFF")

    if head.startswith(b"FUJIFILMCCD-RAW"):
        return Detected("raw", "Fujifilm RAF")
    if head[:4] in (b"II*\x00", b"MM\x00*", b"IIRO", b"IIRS", b"MMOR", b"IIU\x00"):
        return _sniff_tiff(head)
    if head[6:14] == b"HEAPCCDR":
        return Detected("raw", "Canon CRW")
    if head[:4] == b"\x00MRM":
        return Detected("raw", "Minolta MRW")
    if head[:4] == b"FOVb":
        return Detected("raw", "Sigma X3F")  # luego se rechaza: Foveon no es Bayer

    # Imágenes procesadas comunes
    if head[:3] == b"\xff\xd8\xff":
        return _processed("JPEG")
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return _processed("PNG")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return _processed("WebP")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return _processed("GIF")
    if head[:2] == b"BM" and len(head) >= 26:
        return _processed("BMP")
    if head[:4] == b"8BPS":
        return _processed("PSD", "documento de Photoshop")
    if head[:12] == b"\x00\x00\x00\x0cjP  \r\n\x87\n":
        return _processed("JPEG 2000")
    if head[:2] == b"\xff\x0a" or head[:12] == b"\x00\x00\x00\x0cJXL \r\n\x87\n":
        return _processed("JPEG XL")
    if head[:4] == b"%PDF":
        return Detected("processed", "PDF", "es un documento PDF, no una foto RAW.")
    return Detected("maybe_raw", "desconocido")


def sniff(path: str | os.PathLike) -> Detected:
    """Identifica el tipo real de ``path`` leyendo sus primeros bytes."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(HEAD_BYTES)
    except OSError as exc:
        return Detected("empty", "ilegible", f"no se pudo leer el archivo ({exc.strerror or exc}).")
    return sniff_bytes(head)
