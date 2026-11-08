"""Ajustes preestablecidos (presets) y copiar/pegar ajustes. Sin Qt.

Formato ``.fpreset`` (UTF-8 JSON)::

    {"fraw_preset": 1, "name": "Mate", "group": "Creativos", "settings": {"contrast": -0.2, …}}

``settings`` es un subconjunto de claves de :data:`fraw.devparams.SCHEMA`
(valores tal como se guardan en DEV). Aplicar un preset sólo toca sus claves.
Los incorporados viven en ``fraw/gui/presets/*.json`` (mismo formato); los del
usuario en ``<AppData>/presets/*.fpreset``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..devparams import GROUPS, SCHEMA, SPECS, DevParams

__all__ = [
    "Preset", "BUILTIN_DIR", "PRESET_EXT", "load_builtin", "load_dir", "load_preset", "save_preset",
    "apply_settings", "settings_for_groups", "preset_from_params", "keys_for_groups", "safe_filename",
]

BUILTIN_DIR = Path(__file__).resolve().parent / "presets"
PRESET_EXT = ".fpreset"
PRESET_VERSION = 1


@dataclass
class Preset:
    name: str
    group: str = "Usuario"
    settings: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None
    builtin: bool = False

    def to_json(self) -> dict[str, Any]:
        return {"fraw_preset": PRESET_VERSION, "name": self.name, "group": self.group, "settings": dict(self.settings)}

    def groups(self) -> list[str]:
        gs = {SPECS[k].group for k in self.settings if k in SPECS}
        return [g.key for g in GROUPS if g.key in gs]


def _valid_settings(d: Any) -> dict[str, Any]:
    if not isinstance(d, Mapping):
        return {}
    parsed = DevParams.from_dict(d).to_dict(full=True)
    return {k: parsed[k] for k in d if k in SPECS}


def load_preset(path: str | os.PathLike, *, builtin: bool = False) -> Preset:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("settings"), dict):
        raise ValueError(f"{p.name}: no es un preset de CTD Photo Studio")
    if int(data.get("fraw_preset", 1)) > PRESET_VERSION:
        raise ValueError(f"{p.name}: versión de preset no soportada")
    name = str(data.get("name") or p.stem)
    return Preset(name, str(data.get("group") or ("Incorporados" if builtin else "Usuario")),
                  _valid_settings(data["settings"]), p, builtin)


def load_dir(folder: str | os.PathLike, patterns: Iterable[str] = ("*" + PRESET_EXT,), *, builtin: bool = False) -> list[Preset]:
    folder = Path(folder)
    out: list[Preset] = []
    if not folder.is_dir():
        return out
    files = sorted({f for pat in patterns for f in folder.glob(pat)})
    for f in files:
        try:
            out.append(load_preset(f, builtin=builtin))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return out


def load_builtin() -> list[Preset]:
    pres = load_dir(BUILTIN_DIR, ("*.json",), builtin=True)
    order = {name: i for i, name in enumerate(_BUILTIN_ORDER)}
    pres.sort(key=lambda p: (order.get(p.path.stem if p.path else "", 999), p.name))
    return pres


_BUILTIN_ORDER = (
    "01_natural", "02_vivido", "03_retrato_suave", "04_paisaje", "05_bn_alto_contraste", "06_bn_suave",
    "07_pelicula_calida", "08_mate", "09_atardecer", "10_noche_urbana", "11_teal_orange", "12_solo_lente", "13_alto_rango", "14_alta_iso",
)


def safe_filename(name: str) -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().strip(".")
    return s[:80] or "preset"


def save_preset(preset: Preset, folder: str | os.PathLike | None = None, path: str | os.PathLike | None = None) -> Path:
    if path is None:
        if folder is None:
            raise ValueError("folder o path")
        Path(folder).mkdir(parents=True, exist_ok=True)
        path = Path(folder) / (safe_filename(preset.name) + PRESET_EXT)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(preset.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    preset.path = path
    return path


def keys_for_groups(groups: Iterable[str]) -> list[str]:
    gs = set(groups)
    return [s.key for s in SCHEMA if s.group in gs]


def settings_for_groups(params: DevParams, groups: Iterable[str], *, only_changed: bool = False) -> dict[str, Any]:
    keys = keys_for_groups(groups)
    if only_changed:
        keys = [k for k in keys if not params.is_default(k)]
    return params.subset(keys)


def preset_from_params(name: str, params: DevParams, groups: Iterable[str], group_label: str = "Usuario") -> Preset:
    return Preset(name, group_label, settings_for_groups(params, groups))


def apply_settings(params: DevParams, settings: Mapping[str, Any]) -> DevParams:
    """Nuevo DevParams = ``params`` con ``settings`` encima (sólo esas claves cambian)."""
    full = DevParams.coerce(params).to_dict(full=True)
    for k, v in settings.items():
        if k in SPECS:
            full[k] = v
    return DevParams.from_dict(full)
