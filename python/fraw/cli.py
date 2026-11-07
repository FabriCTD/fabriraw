"""``ctd-raw`` — command line tool for FabriRAW (.fraw / .frw / .fset).

Subcommands: ``ingest``, ``burst``, ``info``, ``develop``, ``dev``,
``extract-thumb``, ``verify``. Output is in Spanish.

Exit codes: 0 = OK, 1 = ``verify`` found at least one invalid file,
2 = format error (:class:`~fraw.format.FrawError`), bad arguments or I/O error.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import __version__

__all__ = ["main", "build_parser", "load_any"]

FRAW_EXTS = (".fraw", ".frw")
FSET_EXT = ".fset"
EXPORT_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


class CliError(Exception):
    """Usage / argument error reported as ``error: ...`` with exit code 2."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _out(msg: str = "") -> None:
    print(msg)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _expand(patterns: Sequence[str]) -> list[str]:
    """Expands globs (cmd.exe / PowerShell do not do it on Windows)."""
    out: list[str] = []
    for p in patterns:
        if any(c in p for c in "*?["):
            matches = sorted(glob.glob(p))
            if not matches:
                raise CliError(f"ningún archivo coincide con {p!r}")
            out.extend(matches)
        else:
            out.append(p)
    return out


def _camera_exts() -> frozenset[str]:
    from .ingest import SUPPORTED_EXTENSIONS

    return SUPPORTED_EXTENSIONS


def _magic(path: str | os.PathLike) -> bytes:
    with open(path, "rb") as f:
        return f.read(8)


def load_any(path: str | os.PathLike, frame: int = 0) -> tuple[Any, str]:
    """(FrawFile, kind) for a .fraw/.frw, a frame of a .fset or a camera RAW (in-memory ingest).

    ``kind`` is ``"fraw"``, ``"fset"`` or ``"raw"``.
    """
    from .format import MAGIC, SET_MAGIC, FrawFile
    from .fset import FrawSet

    p = os.fspath(path)
    ext = os.path.splitext(p)[1].lower()
    if ext in _camera_exts():
        from .ingest import ingest

        return ingest(p), "raw"
    m = _magic(p)
    if m == SET_MAGIC:
        s = FrawSet.read(p)
        if not 0 <= frame < len(s.frames):
            raise CliError(f"frame {frame} fuera de rango (el .fset tiene {len(s.frames)})")
        return s.frames[frame], "fset"
    if m != MAGIC and ext not in FRAW_EXTS:
        raise CliError(f"{p}: no es un archivo FabriRAW ni un RAW de cámara soportado")
    return FrawFile.read(p), "fraw"


def _fmt_shutter(s: Any) -> str:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return str(s)
    if v <= 0:
        return str(s)
    if v < 1:
        return f"1/{round(1 / v)} s"
    return f"{v:g} s"


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------
def _chunk_table(data: bytes, magic: bytes) -> list[dict]:
    from .format import _read_chunks

    return [
        {
            "id": c.id.decode("ascii", "replace"),
            "offset": c.offset,
            "length": len(c.payload),
            "critical": c.critical,
        }
        for c in _read_chunks(data, magic)
    ]


def _fraw_summary(f: Any) -> dict:
    h = f.head
    c = f.color
    from .devparams import DevParams

    return {
        "head": {
            "version": f"{h.version_major}.{h.version_minor}",
            "width": h.width,
            "height": h.height,
            "bits_per_sample": h.bits_per_sample,
            "effective_bits": h.effective_bits,
            "cfa_pattern": "".join("RGB"[i] for i in h.cfa_pattern),
            "compression": {0: "ninguna", 1: "zlib"}.get(h.compression, str(h.compression)),
            "orientation": h.orientation,
            "black_level": list(h.black_level),
            "white_level": h.white_level,
            "crop": [h.crop_left, h.crop_top, h.crop_width, h.crop_height],
        },
        "meta": f.meta,
        "colr": {
            "as_shot_wb": [round(v, 6) for v in c.as_shot_wb],
            "daylight_wb": [round(v, 6) for v in c.daylight_wb],
            "rgb_cam": [round(v, 6) for v in c.rgb_cam],
            "cam_xyz": [round(v, 6) for v in c.cam_xyz],
            "illuminant": c.illuminant,
            "flags": c.flags,
        },
        "thumbnail": None
        if f.thumbnail is None
        else {
            "format": {0: "JPEG", 1: "RGB8"}.get(f.thumbnail.format, str(f.thumbnail.format)),
            "width": f.thumbnail.width,
            "height": f.thumbnail.height,
            "bytes": len(f.thumbnail.data),
        },
        "dev": f.dev,
        "dev_effective": DevParams.from_dict(f.dev).to_dict(),
        "extra_chunks": [x.id.decode("ascii", "replace") for x in f.extra_chunks],
    }


def _print_fraw_summary(s: dict, indent: str = "") -> None:
    h = s["head"]
    crop = h["crop"]
    _out(f"{indent}HEAD  v{h['version']}  {h['width']}x{h['height']}  CFA {h['cfa_pattern']}  "
         f"{h['effective_bits']} bits efectivos  compresión {h['compression']}  orientación {h['orientation']}")
    _out(f"{indent}      negro {h['black_level']}  blanco {h['white_level']}  "
         + ("sin recorte" if not any(crop) else f"recorte {crop[2]}x{crop[3]} en ({crop[0]},{crop[1]})"))
    m = s["meta"]
    if m:
        _out(f"{indent}META")
        for k, v in m.items():
            if k == "shutter_s":
                v = f"{v} ({_fmt_shutter(v)})"
            _out(f"{indent}      {k}: {v}")
    else:
        _out(f"{indent}META  (vacío)")
    c = s["colr"]
    _out(f"{indent}COLR  as_shot {c['as_shot_wb']}  daylight {c['daylight_wb']}")
    _out(f"{indent}      rgb_cam {c['rgb_cam']}  flags {c['flags']}  iluminante {c['illuminant']}")
    t = s["thumbnail"]
    _out(f"{indent}THMB  " + ("(no tiene)" if t is None else f"{t['format']} {t['width']}x{t['height']} ({t['bytes']} B)"))
    _out(f"{indent}DEV   " + ("(no tiene; valores por defecto)" if s["dev"] is None else json.dumps(s["dev"], ensure_ascii=False)))
    if s["extra_chunks"]:
        _out(f"{indent}Chunks desconocidos preservados: {', '.join(s['extra_chunks'])}")


def cmd_info(a: argparse.Namespace) -> int:
    from .format import MAGIC, SET_MAGIC, FrawFile
    from .fset import FrawSet

    path = a.file
    with open(path, "rb") as fh:
        data = fh.read()
    size = len(data)
    if data[:8] == SET_MAGIC:
        s = FrawSet.from_bytes(data)
        info = {
            "file": os.fspath(path),
            "type": "fset",
            "size": size,
            "chunks": _chunk_table(data, SET_MAGIC),
            "seth": {"version": 1, "frame_count": len(s.frames), "burst_id": s.burst_id, "created": s.created},
            "frames": [_fraw_summary(f) for f in s.frames],
        }
        if a.json:
            _out(json.dumps(info, ensure_ascii=False, indent=2))
            return 0
        _out(f"{path}: ráfaga FabriRAW (.fset), {size} bytes, {len(s.frames)} frame(s)")
        _out(f"SETH  burst_id {s.burst_id!r}  creado {s.created!r}")
        _out("Chunks:")
        for c in info["chunks"]:
            _out(f"  {c['id']!r:8} offset {c['offset']:>10}  {c['length']:>10} B" + ("  CRITICAL" if c["critical"] else ""))
        for i, fs in enumerate(info["frames"]):
            _out(f"--- Frame {i} ---")
            _print_fraw_summary(fs, "  ")
        return 0
    f = FrawFile.from_bytes(data)
    info = {"file": os.fspath(path), "type": "fraw", "size": size, "chunks": _chunk_table(data, MAGIC), **_fraw_summary(f)}
    if a.json:
        _out(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    _out(f"{path}: FabriRAW (.fraw), {size} bytes")
    _out("Chunks:")
    for c in info["chunks"]:
        _out(f"  {c['id']!r:8} offset {c['offset']:>10}  {c['length']:>10} B" + ("  CRITICAL" if c["critical"] else ""))
    _print_fraw_summary(info)
    return 0


# ---------------------------------------------------------------------------
# ingest / burst
# ---------------------------------------------------------------------------
def cmd_ingest(a: argparse.Namespace) -> int:
    from .ingest import ingest_to_file

    srcs = _expand(a.src)
    if a.output and len(srcs) > 1:
        raise CliError("-o sólo sirve con un único archivo de entrada; usá --outdir para varios")
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
    for src in srcs:
        if a.output:
            dst = a.output
        else:
            base = Path(src).with_suffix(".fraw").name
            dst = os.path.join(a.outdir, base) if a.outdir else os.fspath(Path(src).with_suffix(".fraw"))
        f = ingest_to_file(src, dst, compress=not a.no_compress, thumbnail=not a.no_thumb)
        cam = " ".join(str(f.meta.get(k, "")) for k in ("make", "model")).strip() or "cámara desconocida"
        _out(f"{src} -> {dst}  ({f.head.width}x{f.head.height}, {cam}, {os.path.getsize(dst)} bytes)")
    return 0


def cmd_burst(a: argparse.Namespace) -> int:
    from .ingest import ingest_burst

    srcs = _expand(a.src)
    s = ingest_burst(srcs, a.output, compress=not a.no_compress, thumbnail=not a.no_thumb)
    _out(f"{len(s.frames)} frame(s) -> {a.output}  (burst_id {s.burst_id}, {os.path.getsize(a.output)} bytes)")
    return 0


# ---------------------------------------------------------------------------
# develop
# ---------------------------------------------------------------------------
def _parse_wb(v: str) -> dict:
    v = v.strip().lower()
    if v in ("as_shot", "asshot", "camera"):
        return {"wb_mode": "as_shot"}
    if v in ("daylight", "dia", "día"):
        return {"wb_mode": "daylight"}
    parts = v.replace(";", ",").split(",")
    if len(parts) == 3:
        try:
            vals = [float(x) for x in parts]
        except ValueError:
            vals = []
        if len(vals) == 3 and all(x > 0 for x in vals):
            return {"wb_mode": "custom", "wb_custom": vals}
    raise CliError(f"--wb inválido {v!r}: usá as_shot, daylight o R,G,B (p. ej. 2.0,1.0,1.5)")


def _parse_gamma(v: str) -> dict:
    v = v.strip().lower()
    if v in ("srgb", "linear"):
        return {"gamma_mode": v}
    if v.startswith("power"):
        g = 2.2
        if ":" in v:
            try:
                g = float(v.split(":", 1)[1])
            except ValueError:
                raise CliError(f"--gamma inválido {v!r}") from None
        return {"gamma_mode": "power", "gamma": g}
    raise CliError(f"--gamma inválido {v!r}: usá srgb, power:2.2 o linear")


def cmd_develop(a: argparse.Namespace) -> int:
    from .devparams import DevParams
    from .format import update_dev
    from .processor import FRAWProcessor

    out = Path(a.output)
    ext = out.suffix.lower()
    if ext not in EXPORT_EXTS:
        raise CliError(f"formato de salida no soportado {ext or '(sin extensión)'!r}: usá .jpg, .png o .tif")
    if ext in (".jpg", ".jpeg") and a.bits == 16:
        raise CliError("JPEG sólo admite 8 bits; usá .png o .tif para 16 bits")
    f, kind = load_any(a.file, a.frame)
    d = DevParams.from_dict(f.dev).to_dict()
    for name in ("ev", "temp", "tint", "contrast", "saturation", "highlights", "shadows"):
        val = getattr(a, name)
        if val is not None:
            key = {"ev": "exposure_ev", "temp": "temperature_shift", "tint": "tint_shift"}.get(name, name)
            d[key] = val
    if a.wb:
        d.update(_parse_wb(a.wb))
    if a.gamma:
        d.update(_parse_gamma(a.gamma))
    if a.demosaic:
        d["demosaic"] = a.demosaic
    params = DevParams.from_dict(d)
    proc = FRAWProcessor(f)
    proc.export(out, params, bits=a.bits, quality=a.quality, half_size=a.half)
    _out(f"{a.file} -> {out}  ({a.bits} bits, demosaico {'superpíxel 1/2' if a.half else params.demosaic})")
    if a.save_dev:
        if kind != "fraw":
            raise CliError("--save-dev sólo funciona con archivos .fraw/.frw")
        update_dev(a.file, params.to_dict())
        _out(f"DEV guardado en {a.file}")
    return 0


# ---------------------------------------------------------------------------
# dev
# ---------------------------------------------------------------------------
def _parse_set_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        pass
    if "," in raw:
        try:
            return [float(x) for x in raw.split(",")]
        except ValueError:
            pass
    return raw


def cmd_dev(a: argparse.Namespace) -> int:
    from .devparams import DevParams
    from .format import FrawFile, update_dev

    if _magic(a.file) != b"FRAW\r\n\x1a\n":
        raise CliError("dev sólo funciona con archivos .fraw/.frw (no .fset ni RAW de cámara)")
    f = FrawFile.read(a.file)
    if a.reset:
        update_dev(a.file, None)
        _out(f"DEV eliminado de {a.file} (vuelven los valores por defecto)")
        return 0
    if a.set:
        known = set(DevParams().to_dict())
        d = dict(f.dev or {})
        for item in a.set:
            if "=" not in item:
                raise CliError(f"--set espera clave=valor, recibido {item!r}")
            k, v = item.split("=", 1)
            k = k.strip()
            if k not in known or k == "version":
                raise CliError(f"clave DEV desconocida {k!r}; válidas: {', '.join(sorted(known - {'version'}))}")
            d[k] = _parse_set_value(v.strip())
        new = DevParams.from_dict(d).to_dict()
        update_dev(a.file, new)
        _out(f"DEV actualizado en {a.file}:")
        _out(json.dumps(new, ensure_ascii=False, indent=2))
        return 0
    # --show (default)
    if f.dev is None:
        _out("DEV: (no tiene; se usan los valores por defecto)")
    else:
        _out("DEV guardado:")
        _out(json.dumps(f.dev, ensure_ascii=False, indent=2))
    _out("Valores efectivos:")
    _out(json.dumps(DevParams.from_dict(f.dev).to_dict(), ensure_ascii=False, indent=2))
    return 0


# ---------------------------------------------------------------------------
# extract-thumb / verify
# ---------------------------------------------------------------------------
def cmd_extract_thumb(a: argparse.Namespace) -> int:
    f, _kind = load_any(a.file, a.frame)
    t = f.thumbnail
    if t is None:
        raise CliError(f"{a.file} no tiene miniatura (THMB)")
    out = Path(a.output)
    ext = out.suffix.lower()
    if t.format == 0 and ext in (".jpg", ".jpeg"):
        out.write_bytes(t.data)
    else:
        from PIL import Image

        if t.format == 0:
            im = Image.open(io.BytesIO(t.data))
            im.load()
        else:
            im = Image.frombytes("RGB", (t.width, t.height), t.data)
        im.convert("RGB").save(out)
    _out(f"Miniatura {t.width}x{t.height} -> {out}")
    return 0


def _verify_one(path: str) -> str | None:
    """None if valid, else an error message."""
    from .format import SET_MAGIC, FrawError, FrawFile
    from .fset import FrawSet

    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data[:8] == SET_MAGIC:
            s = FrawSet.from_bytes(data)
            return None if s.frames else "sin frames"
        FrawFile.from_bytes(data)
        return None
    except FrawError as e:
        return str(e)
    except OSError as e:
        return f"error de E/S: {e.strerror or e}"


def cmd_verify(a: argparse.Namespace) -> int:
    files = _expand(a.files)
    bad = 0
    for p in files:
        msg = _verify_one(p)
        if msg is None:
            _out(f"OK     {p}")
        else:
            bad += 1
            _out(f"ERROR  {p}: {msg}")
    _out(f"{len(files) - bad}/{len(files)} archivo(s) válidos")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ctd-raw",
        description="CTD Photo Studio — herramienta de línea de comandos para FabriRAW (.fraw/.frw/.fset).",
        epilog="https://fraw.fabrictd.com",
    )
    p.add_argument("--version", action="version", version=f"ctd-raw {__version__} (FabriRAW 1.0)")
    sub = p.add_subparsers(dest="cmd", metavar="COMANDO")
    sub.required = True

    s = sub.add_parser("ingest", help="convierte RAW de cámara a .fraw")
    s.add_argument("src", nargs="+", help="archivos RAW (acepta comodines)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("-o", "--output", help="archivo .fraw de salida (sólo con una entrada)")
    g.add_argument("--outdir", help="carpeta de salida (por defecto, junto al original)")
    s.add_argument("--no-compress", action="store_true", help="DATA sin comprimir")
    s.add_argument("--no-thumb", action="store_true", help="no generar miniatura THMB")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("burst", help="empaqueta varios RAW en una ráfaga .fset")
    s.add_argument("src", nargs="+", help="archivos RAW (acepta comodines)")
    s.add_argument("-o", "--output", required=True, help="archivo .fset de salida")
    s.add_argument("--no-compress", action="store_true")
    s.add_argument("--no-thumb", action="store_true")
    s.set_defaults(func=cmd_burst)

    s = sub.add_parser("info", help="muestra chunks, HEAD, META, COLR y DEV")
    s.add_argument("file")
    s.add_argument("--json", action="store_true", help="salida JSON")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("develop", help="revela a JPEG/PNG/TIFF")
    s.add_argument("file", help=".fraw/.frw/.fset o RAW de cámara")
    s.add_argument("-o", "--output", required=True, help="salida .jpg/.png/.tif")
    s.add_argument("--frame", type=int, default=0, help="frame del .fset (por defecto 0)")
    s.add_argument("--ev", type=float, help="exposición en EV [-4, 4]")
    s.add_argument("--wb", help="as_shot | daylight | R,G,B")
    s.add_argument("--temp", type=float, help="temperatura [-1, 1]")
    s.add_argument("--tint", type=float, help="tinte [-1, 1]")
    s.add_argument("--contrast", type=float, help="contraste [-1, 1]")
    s.add_argument("--saturation", type=float, help="saturación [0, 2]")
    s.add_argument("--highlights", type=float, help="altas luces [-1, 1]")
    s.add_argument("--shadows", type=float, help="sombras [-1, 1]")
    s.add_argument("--gamma", help="srgb | power:2.2 | linear")
    s.add_argument("--demosaic", choices=("ahd", "bilinear"))
    s.add_argument("--bits", type=int, choices=(8, 16), default=8)
    s.add_argument("--quality", type=int, default=95, help="calidad JPEG (1-100)")
    s.add_argument("--half", action="store_true", help="mitad de resolución (superpíxel, rápido)")
    s.add_argument("--save-dev", action="store_true", help="guarda los parámetros usados en el DEV del archivo")
    s.set_defaults(func=cmd_develop)

    s = sub.add_parser("dev", help="muestra o edita el revelado no destructivo (DEV)")
    s.add_argument("file")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--set", nargs="+", metavar="CLAVE=VALOR", help="p. ej. exposure_ev=0.7 wb_mode=daylight")
    g.add_argument("--reset", action="store_true", help="elimina el DEV")
    g.add_argument("--show", action="store_true", help="muestra el DEV (por defecto)")
    s.set_defaults(func=cmd_dev)

    s = sub.add_parser("extract-thumb", help="extrae la miniatura THMB")
    s.add_argument("file")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--frame", type=int, default=0)
    s.set_defaults(func=cmd_extract_thumb)

    s = sub.add_parser("verify", help="valida estructura y CRC (exit 1 si alguno falla)")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_verify)
    return p


def _setup_streams() -> None:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _setup_streams()
    parser = build_parser()
    try:
        a = parser.parse_args(argv)
    except SystemExit as e:  # --help / --version (0) or usage error (2)
        return int(e.code or 0)
    from .format import FrawError

    try:
        return int(a.func(a))
    except FrawError as e:
        _err(f"error: archivo FabriRAW inválido ({e.code.value}): {e.msg}")
        return 2
    except CliError as e:
        _err(f"error: {e}")
        return 2
    except FileNotFoundError as e:
        _err(f"error: no existe el archivo {e.filename or e}")
        return 2
    except OSError as e:
        _err(f"error de E/S: {e}")
        return 2
    except ImportError as e:
        _err(f"error: falta una dependencia opcional: {e}")
        return 2
    except ValueError as e:
        _err(f"error: {e}")
        return 2
    except Exception as e:  # LibRaw errors, etc.
        if type(e).__module__.startswith("rawpy"):
            _err(f"error: LibRaw no pudo leer el RAW: {e}")
            return 2
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
