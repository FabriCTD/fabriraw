"""Exportación por lotes (Qt-free): plantilla de nombre, destino y bucle con progreso/cancelación."""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..devparams import DevParams

__all__ = ["ExportOptions", "ExportJob", "render_name", "unique_path", "export_batch", "FORMAT_EXT"]

FORMAT_EXT = {"JPEG": ".jpg", "PNG": ".png", "TIFF": ".tif"}


@dataclass
class ExportOptions:
    fmt: str = "JPEG"  # JPEG | PNG | TIFF
    bits: int = 8
    quality: int = 92
    long_edge: int | None = None
    output_sharpen: str = "none"  # none | screen | matte | glossy
    metadata: bool = True
    embed_icc: bool = True
    template: str = "{nombre}"
    folder: str = ""
    demosaic: str | None = None
    open_folder: bool = False

    @property
    def ext(self) -> str:
        return FORMAT_EXT.get(self.fmt.upper(), ".jpg")


@dataclass
class ExportJob:
    source: Path  # .fraw/.frw/.fset de origen (para el nombre)
    params: DevParams | None = None  # None → DEV guardado en el archivo
    frawfile: Any = None  # FrawFile ya cargado (opcional)
    meta: dict = field(default_factory=dict)


_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def render_name(template: str, source: Path, n: int, meta: dict | None = None, total: int = 1) -> str:
    """``{nombre}`` (archivo sin extensión), ``{fecha}`` (AAAAMMDD de la toma o de hoy),
    ``{n}`` (número de orden, 3 dígitos o más), ``{camara}``, ``{iso}``."""
    meta = meta or {}
    dt = str(meta.get("datetime") or "")
    m = re.match(r"(\d{4})[-:](\d{2})[-:](\d{2})", dt)
    fecha = "".join(m.groups()) if m else _dt.date.today().strftime("%Y%m%d")
    width = max(3, len(str(total)))
    vals = {
        "nombre": source.stem,
        "fecha": fecha,
        "n": str(n).zfill(width),
        "camara": " ".join(str(meta.get(k, "")) for k in ("make", "model")).strip() or "camara",
        "iso": str(meta.get("iso", "")),
    }
    out = template or "{nombre}"
    for k, v in vals.items():
        out = out.replace("{" + k + "}", v)
    out = _BAD.sub("_", out).strip().strip(".")
    return out or source.stem


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    i = 2
    while True:
        q = p.with_name(f"{p.stem}-{i}{p.suffix}")
        if not q.exists():
            return q
        i += 1


def export_batch(jobs: list[ExportJob], opts: ExportOptions,
                 report: Callable[[int, str], None] = lambda i, t: None,
                 cancelled: Callable[[], bool] = lambda: False) -> list[Path]:
    """Exporta cada trabajo. Devuelve las rutas escritas (se detiene al cancelar)."""
    from ..format import FrawFile  # noqa: PLC0415
    from ..processor import FRAWProcessor  # noqa: PLC0415
    from .engine import Engine  # noqa: PLC0415

    out: list[Path] = []
    folder = Path(opts.folder) if opts.folder else None
    total = len(jobs)
    for i, job in enumerate(jobs):
        if cancelled():
            break
        report(i, f"Exportando {job.source.name} ({i + 1}/{total})…")
        f = job.frawfile
        if f is None:
            with open(job.source, "rb") as fh:
                magic = fh.read(8)
            if magic.startswith(b"FSET"):
                from ..fset import FrawSet  # noqa: PLC0415

                f = FrawSet.read(job.source).frames[0]
            else:
                f = FrawFile.read(job.source)
        params = job.params if job.params is not None else DevParams.from_dict(f.dev)
        meta = dict(f.meta or {}) | dict(job.meta or {})
        dest_dir = folder or job.source.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = unique_path(dest_dir / (render_name(opts.template, job.source, i + 1, meta, total) + opts.ext))
        eng = Engine(FRAWProcessor(f))
        bits = 8 if opts.fmt.upper() == "JPEG" else int(opts.bits)
        eng.export(dest, params, bits=bits, quality=opts.quality, long_edge=opts.long_edge,
                   output_sharpen=opts.output_sharpen, embed_icc=opts.embed_icc, metadata=opts.metadata,
                   demosaic=opts.demosaic or params.demosaic, meta=meta)
        out.append(dest)
    report(total, "Listo")
    return out
