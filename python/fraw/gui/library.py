"""Biblioteca: calificación (0–5 estrellas) y marca de color en el chunk META. Sin Qt.

``META`` va antes de ``DATA`` (SPEC §3), así que cambiarlo exige reescribir el
archivo. :func:`update_meta` lo hace *a nivel de bytes*: valida el framing y los
CRC, reemplaza sólo el payload de META (o lo inserta tras HEAD si no existía) y
copia todos los demás chunks tal cual — DATA no se decodifica ni se recomprime.
Escribe a un temporal en la misma carpeta y hace ``os.replace`` (atómico).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from ..format import MAGIC, _canonical_flags, _frame, _json_object_from_payload, _json_object_to_payload, _read_chunks

__all__ = ["update_meta", "read_meta", "set_rating", "set_label", "LABELS", "LABEL_COLORS", "stars"]

LABELS = (("", "Sin marca"), ("red", "Rojo"), ("yellow", "Amarillo"), ("green", "Verde"),
          ("blue", "Azul"), ("purple", "Púrpura"))
LABEL_COLORS = {"red": "#E5484D", "yellow": "#F0B232", "green": "#23A559", "blue": "#4C8DFF", "purple": "#A86FDF"}
ID_META = b"META"
ID_HEAD = b"HEAD"


def stars(n: int) -> str:
    n = max(0, min(5, int(n or 0)))
    return "★" * n + "☆" * (5 - n)


def read_meta(path: str | os.PathLike) -> dict[str, Any]:
    from .thumbs import peek_file  # noqa: PLC0415

    return dict(peek_file(path).meta or {})


def update_meta(path: str | os.PathLike, updates: Mapping[str, Any]) -> dict[str, Any]:
    """Fusiona ``updates`` en META (valor ``None`` = borrar la clave). Devuelve el META nuevo."""
    path = Path(path)
    data = path.read_bytes()
    chunks = _read_chunks(data, MAGIC)  # framing + CRC de todo el archivo
    meta: dict[str, Any] = {}
    idx = next((i for i, c in enumerate(chunks) if c.id == ID_META), None)
    if idx is not None:
        meta = _json_object_from_payload(chunks[idx].payload, "META")
    for k, v in updates.items():
        if v is None:
            meta.pop(k, None)
        else:
            meta[k] = v
    new_chunk = _frame(ID_META, _canonical_flags(ID_META), _json_object_to_payload(meta, "META"))
    parts: list[bytes | memoryview] = [MAGIC]
    mv = memoryview(data)
    for i, c in enumerate(chunks):
        if i == idx:
            parts.append(new_chunk)
            continue
        parts.append(mv[c.offset : c.offset + c.total_size])
        if idx is None and c.id == ID_HEAD:
            parts.append(new_chunk)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as f:
            for part in parts:
                f.write(part)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return meta


def set_rating(path: str | os.PathLike, rating: int) -> dict[str, Any]:
    r = max(0, min(5, int(rating)))
    return update_meta(path, {"rating": r if r else None})


def set_label(path: str | os.PathLike, label: str) -> dict[str, Any]:
    lab = str(label or "")
    if lab and lab not in LABEL_COLORS:
        raise ValueError(f"marca desconocida: {lab}")
    return update_meta(path, {"label": lab or None})

