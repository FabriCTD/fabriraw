"""Parámetros de revelado no destructivo (SPEC §4.6, chunk ``DEV ``) — versión 2.

Toda la información de cada ajuste vive en :data:`SCHEMA` (clave, nombre visible,
categoría, tipo, rango, valor por defecto, paso, decimales y unidad). Es el
**contrato único** entre el archivo, el motor (:mod:`fraw.processor`) y la
interfaz (que genera sus paneles a partir del esquema).

Compatibilidad:

* Lee DEV v1 (las 12 claves originales conservan nombre, rango y defecto).
* El parseo es tolerante y nunca lanza: claves ausentes → defecto, números fuera
  de rango → recortados, tipos inválidos (incluido ``bool`` en numéricos y NaN/inf)
  → defecto, claves desconocidas → ignoradas.
* Un lector v1 que encuentre un DEV v2 ignora las claves nuevas (SPEC §4.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping

__all__ = [
    "DevParams", "ParamSpec", "Group", "SCHEMA", "GROUPS", "SPECS", "DEV_VERSION",
    "RANGES", "WB_MODES", "GAMMA_MODES", "DEMOSAIC_METHODS", "HIGHLIGHT_MODES",
    "HSL_BANDS", "CURVE_KEYS", "IDENTITY_CURVE", "normalize_curve", "is_identity_curve",
]

DEV_VERSION = 2

WB_MODES = ("as_shot", "daylight", "custom", "kelvin")
GAMMA_MODES = ("srgb", "power", "linear")
DEMOSAIC_METHODS = ("bilinear", "ahd")
HIGHLIGHT_MODES = ("clip", "blend", "reconstruct")
CROP_ASPECTS = ("free", "original", "1:1", "4:5", "5:4", "3:2", "2:3", "16:9", "9:16", "4:3", "3:4")
HSL_BANDS = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
HSL_BAND_LABELS = {
    "red": "Rojo", "orange": "Naranja", "yellow": "Amarillo", "green": "Verde",
    "aqua": "Aguamarina", "blue": "Azul", "purple": "Púrpura", "magenta": "Magenta",
}
#: Centro de cada banda HSL en grados de tono (HSV/HSL estándar).
HSL_BAND_HUES = {
    "red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
    "aqua": 180.0, "blue": 225.0, "purple": 270.0, "magenta": 315.0,
}
CURVE_KEYS = ("curve_master", "curve_red", "curve_green", "curve_blue")
IDENTITY_CURVE: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
CURVE_MAX_POINTS = 16

WB_CUSTOM_RANGE = (1e-3, 64.0)
_DEFAULT_WB_CUSTOM = (2.0, 1.0, 1.5)


# --------------------------------------------------------------------- schema
@dataclass(frozen=True)
class ParamSpec:
    """Descripción de un ajuste.

    ``kind``: ``"float"`` | ``"int"`` | ``"bool"`` | ``"enum"`` | ``"curve"`` | ``"wb3"``.
    ``ui_scale``: factor de visualización (p. ej. 100 → el control muestra −100…+100
    mientras el archivo guarda −1…+1). ``hidden``: no se muestra como control propio.
    """

    key: str
    label: str
    group: str
    kind: str
    default: Any
    lo: float | None = None
    hi: float | None = None
    step: float = 0.01
    decimals: int = 2
    unit: str = ""
    ui_scale: float = 1.0
    choices: tuple[tuple[str, str], ...] = ()
    section: str = ""
    hidden: bool = False
    tooltip: str = ""


@dataclass(frozen=True)
class Group:
    key: str
    label: str
    icon: str = ""


GROUPS: tuple[Group, ...] = (
    Group("basic", "Básico", "sun"),
    Group("tone_curve", "Curva de tonos", "curve"),
    Group("hsl", "Mezclador de color (HSL)", "palette"),
    Group("grading", "Gradación de color", "wheel"),
    Group("detail", "Detalle", "sharp"),
    Group("optics", "Óptica", "lens"),
    Group("geometry", "Geometría y recorte", "crop"),
    Group("effects", "Efectos", "fx"),
    Group("calibration", "Calibración", "cal"),
    Group("process", "Proceso", "gear"),
)


def _f(key, label, group, lo, hi, default=0.0, *, step=0.01, decimals=2, unit="", ui_scale=1.0,
       section="", tooltip="", hidden=False) -> ParamSpec:
    return ParamSpec(key, label, group, "float", float(default), float(lo), float(hi), step, decimals,
                     unit, ui_scale, (), section, hidden, tooltip)


def _pct(key, label, group, lo=-1.0, hi=1.0, default=0.0, *, section="", tooltip="") -> ParamSpec:
    """Ajuste −1…+1 (o 0…1) mostrado como −100…+100 en la interfaz."""
    return _f(key, label, group, lo, hi, default, step=0.01, decimals=0, ui_scale=100.0,
              section=section, tooltip=tooltip)


def _e(key, label, group, choices, default, *, section="", tooltip="") -> ParamSpec:
    return ParamSpec(key, label, group, "enum", default, choices=tuple(choices), section=section, tooltip=tooltip)


def _b(key, label, group, default=False, *, section="") -> ParamSpec:
    return ParamSpec(key, label, group, "bool", bool(default), section=section)


def _build_schema() -> tuple[ParamSpec, ...]:
    s: list[ParamSpec] = [
        # ---------------------------------------------------------------- Básico
        _e("wb_mode", "Balance de blancos", "basic",
           (("as_shot", "Como se tomó"), ("daylight", "Luz día"), ("kelvin", "Temperatura (K)"),
            ("custom", "Personalizado / cuentagotas")), "as_shot", section="Balance de blancos"),
        _f("wb_kelvin", "Temperatura", "basic", 2000, 50000, 5500, step=50, decimals=0, unit="K",
           section="Balance de blancos", tooltip="Sólo en modo Temperatura (K)"),
        _f("wb_kelvin_tint", "Matiz", "basic", -150, 150, 0, step=1, decimals=0,
           section="Balance de blancos", tooltip="Sólo en modo Temperatura (K): − verde / + magenta"),
        ParamSpec("wb_custom", "Multiplicadores WB", "basic", "wb3", _DEFAULT_WB_CUSTOM,
                  WB_CUSTOM_RANGE[0], WB_CUSTOM_RANGE[1], 0.001, 3, hidden=True,
                  section="Balance de blancos"),
        _pct("temperature_shift", "Temperatura (ajuste fino)", "basic", section="Balance de blancos",
             tooltip="− frío / + cálido, sobre el balance elegido"),
        _pct("tint_shift", "Matiz (ajuste fino)", "basic", section="Balance de blancos",
             tooltip="− verde / + magenta, sobre el balance elegido"),
        _f("exposure_ev", "Exposición", "basic", -4, 4, 0, step=0.01, decimals=2, unit="EV", section="Tono"),
        _pct("contrast", "Contraste", "basic", section="Tono"),
        _pct("highlights", "Altas luces", "basic", section="Tono"),
        _pct("shadows", "Sombras", "basic", section="Tono"),
        _pct("whites", "Blancos", "basic", section="Tono"),
        _pct("blacks", "Negros", "basic", section="Tono"),
        _pct("texture", "Textura", "basic", section="Presencia", tooltip="Detalle de frecuencias medias"),
        _pct("clarity", "Claridad", "basic", section="Presencia", tooltip="Contraste local"),
        _pct("dehaze", "Neblina", "basic", section="Presencia", tooltip="+ quita neblina / − agrega"),
        _pct("vibrance", "Intensidad", "basic", section="Presencia",
             tooltip="Saturación que protege tonos de piel y colores ya saturados"),
        _f("saturation", "Saturación", "basic", 0, 2, 1, step=0.01, decimals=0, ui_scale=100.0,
           section="Presencia", tooltip="100 = sin cambio"),
        # ---------------------------------------------------------- Curva de tonos
        _pct("curve_p_highlights", "Iluminaciones", "tone_curve", section="Curva paramétrica"),
        _pct("curve_p_lights", "Claros", "tone_curve", section="Curva paramétrica"),
        _pct("curve_p_darks", "Oscuros", "tone_curve", section="Curva paramétrica"),
        _pct("curve_p_shadows", "Sombras", "tone_curve", section="Curva paramétrica"),
    ]
    for k, lbl in zip(CURVE_KEYS, ("Curva RGB", "Curva rojo", "Curva verde", "Curva azul")):
        s.append(ParamSpec(k, lbl, "tone_curve", "curve", IDENTITY_CURVE, 0.0, 1.0, section="Curva de puntos"))
    # ------------------------------------------------------------------ HSL
    for comp, comp_lbl in (("h", "Tono"), ("s", "Saturación"), ("l", "Luminancia")):
        for band in HSL_BANDS:
            s.append(_pct(f"hsl_{band}_{comp}", HSL_BAND_LABELS[band], "hsl", section=comp_lbl))
    # ------------------------------------------------------------- Gradación
    for zone, zlbl in (("shadows", "Sombras"), ("midtones", "Medios tonos"),
                       ("highlights", "Iluminaciones"), ("global", "Global")):
        s.append(_f(f"grade_{zone}_hue", "Tono", "grading", 0, 360, 0, step=1, decimals=0, unit="°", section=zlbl))
        s.append(_pct(f"grade_{zone}_sat", "Saturación", "grading", 0.0, 1.0, 0.0, section=zlbl))
        s.append(_pct(f"grade_{zone}_lum", "Luminancia", "grading", section=zlbl))
    s += [
        _pct("grade_blending", "Fusión", "grading", 0.0, 1.0, 0.5, section="Ajustes"),
        _pct("grade_balance", "Equilibrio", "grading", section="Ajustes"),
        # ------------------------------------------------------------- Detalle
        _f("sharpen_amount", "Cantidad", "detail", 0, 1.5, 0.0, step=0.01, decimals=0, ui_scale=100.0,
           section="Enfoque"),
        _f("sharpen_radius", "Radio", "detail", 0.5, 3.0, 1.0, step=0.1, decimals=1, unit="px", section="Enfoque"),
        _pct("sharpen_detail", "Detalle", "detail", 0.0, 1.0, 0.25, section="Enfoque"),
        _pct("sharpen_masking", "Máscara", "detail", 0.0, 1.0, 0.0, section="Enfoque",
             tooltip="Limita el enfoque a los bordes"),
        _pct("nr_luminance", "Luminancia", "detail", 0.0, 1.0, 0.0, section="Reducción de ruido"),
        _pct("nr_luminance_detail", "Detalle de luminancia", "detail", 0.0, 1.0, 0.5, section="Reducción de ruido"),
        _pct("nr_color", "Color", "detail", 0.0, 1.0, 0.0, section="Reducción de ruido"),
        _pct("nr_color_detail", "Detalle de color", "detail", 0.0, 1.0, 0.5, section="Reducción de ruido"),
        # -------------------------------------------------------------- Óptica
        _pct("lens_distortion", "Distorsión", "optics", section="Corrección de lente",
             tooltip="+ corrige barril / − corrige cojín"),
        _pct("lens_vignette", "Viñeteo de lente", "optics", section="Corrección de lente",
             tooltip="+ aclara las esquinas"),
        _pct("ca_red_cyan", "Aberración rojo/cian", "optics", section="Aberración cromática"),
        _pct("ca_blue_yellow", "Aberración azul/amarillo", "optics", section="Aberración cromática"),
        _pct("defringe", "Eliminar halos", "optics", 0.0, 1.0, 0.0, section="Aberración cromática"),
        # ----------------------------------------------------------- Geometría
        _e("rotate_90", "Girar", "geometry", (("0", "0°"), ("90", "90° horario"), ("180", "180°"),
                                               ("270", "90° antihorario")), "0", section="Orientación"),
        _b("flip_h", "Voltear horizontal", "geometry", section="Orientación"),
        _b("flip_v", "Voltear vertical", "geometry", section="Orientación"),
        _f("rotate_deg", "Enderezar", "geometry", -45, 45, 0, step=0.05, decimals=2, unit="°", section="Enderezar"),
        _pct("perspective_vertical", "Perspectiva vertical", "geometry", section="Transformar"),
        _pct("perspective_horizontal", "Perspectiva horizontal", "geometry", section="Transformar"),
        _f("perspective_scale", "Escala", "geometry", 0.5, 1.5, 1.0, step=0.01, decimals=0, ui_scale=100.0,
           unit="%", section="Transformar"),
        _e("crop_aspect", "Proporción", "geometry", tuple((a, {"free": "Libre", "original": "Original"}.get(a, a))
                                                         for a in CROP_ASPECTS), "free", section="Recorte"),
        _f("crop_left", "Izquierda", "geometry", 0, 1, 0, step=0.001, decimals=1, ui_scale=100.0, unit="%",
           section="Recorte"),
        _f("crop_top", "Arriba", "geometry", 0, 1, 0, step=0.001, decimals=1, ui_scale=100.0, unit="%",
           section="Recorte"),
        _f("crop_right", "Derecha", "geometry", 0, 1, 1, step=0.001, decimals=1, ui_scale=100.0, unit="%",
           section="Recorte"),
        _f("crop_bottom", "Abajo", "geometry", 0, 1, 1, step=0.001, decimals=1, ui_scale=100.0, unit="%",
           section="Recorte"),
        # ------------------------------------------------------------- Efectos
        _pct("vignette_amount", "Cantidad", "effects", section="Viñeta tras recorte"),
        _pct("vignette_midpoint", "Punto medio", "effects", 0.0, 1.0, 0.5, section="Viñeta tras recorte"),
        _pct("vignette_roundness", "Redondez", "effects", section="Viñeta tras recorte"),
        _pct("vignette_feather", "Difuminado", "effects", 0.0, 1.0, 0.5, section="Viñeta tras recorte"),
        _pct("vignette_highlights", "Altas luces", "effects", 0.0, 1.0, 0.0, section="Viñeta tras recorte",
             tooltip="Protege las altas luces de la viñeta oscura"),
        _pct("grain_amount", "Cantidad", "effects", 0.0, 1.0, 0.0, section="Grano"),
        _pct("grain_size", "Tamaño", "effects", 0.0, 1.0, 0.25, section="Grano"),
        _pct("grain_roughness", "Aspereza", "effects", 0.0, 1.0, 0.5, section="Grano"),
        # ---------------------------------------------------------- Calibración
        _pct("cal_shadows_tint", "Matiz de sombras", "calibration", section="Sombras"),
        _pct("cal_red_hue", "Tono", "calibration", section="Primario rojo"),
        _pct("cal_red_sat", "Saturación", "calibration", section="Primario rojo"),
        _pct("cal_green_hue", "Tono", "calibration", section="Primario verde"),
        _pct("cal_green_sat", "Saturación", "calibration", section="Primario verde"),
        _pct("cal_blue_hue", "Tono", "calibration", section="Primario azul"),
        _pct("cal_blue_sat", "Saturación", "calibration", section="Primario azul"),
        # -------------------------------------------------------------- Proceso
        _e("demosaic", "Demosaico", "process", (("ahd", "AHD (calidad)"), ("bilinear", "Bilineal (rápido)")),
           "ahd", section="Proceso"),
        _e("highlight_mode", "Altas luces quemadas", "process",
           (("clip", "Recortar"), ("blend", "Mezclar"), ("reconstruct", "Reconstruir")), "clip", section="Proceso"),
        _e("gamma_mode", "Curva de salida", "process",
           (("srgb", "sRGB"), ("power", "Potencia"), ("linear", "Lineal")), "srgb", section="Proceso"),
        _f("gamma", "Gamma", "process", 1.0, 4.0, 2.2, step=0.01, decimals=2, section="Proceso",
           tooltip="Sólo con curva de salida Potencia"),
    ]
    return tuple(s)


SCHEMA: tuple[ParamSpec, ...] = _build_schema()
SPECS: dict[str, ParamSpec] = {p.key: p for p in SCHEMA}
assert len(SPECS) == len(SCHEMA), "claves duplicadas en SCHEMA"

#: Rangos numéricos (compatibilidad con v1 y validación rápida).
RANGES: dict[str, tuple[float, float]] = {
    p.key: (p.lo, p.hi) for p in SCHEMA if p.kind in ("float", "int")  # type: ignore[misc]
}


# --------------------------------------------------------------------- helpers
def _num(value: Any) -> float | None:
    """Número real finito (no bool) → float; si no, None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        v = float(value)
    except (OverflowError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def normalize_curve(points: Any) -> tuple[tuple[float, float], ...] | None:
    """Valida una curva ``[[x, y], …]``: 2–16 puntos en [0, 1], x estrictamente creciente.

    Devuelve ``None`` si no es una curva válida (se usa el defecto).
    """
    if not isinstance(points, (list, tuple)) or not 2 <= len(points) <= CURVE_MAX_POINTS:
        return None
    out: list[tuple[float, float]] = []
    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            return None
        x, y = _num(pt[0]), _num(pt[1])
        if x is None or y is None:
            return None
        out.append((_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0)))
    out.sort(key=lambda p: p[0])
    dedup: list[tuple[float, float]] = []
    for x, y in out:
        if dedup and x - dedup[-1][0] < 1e-4:
            dedup[-1] = (dedup[-1][0], y)  # x repetida: gana la última
        else:
            dedup.append((x, y))
    return tuple(dedup) if len(dedup) >= 2 else None


def is_identity_curve(points: Iterable[Iterable[float]]) -> bool:
    pts = [tuple(p) for p in points]
    return all(abs(x - y) < 1e-6 for x, y in pts) and pts[0][0] <= 1e-6 and pts[-1][0] >= 1 - 1e-6


def _parse_value(spec: ParamSpec, raw: Any) -> tuple[bool, Any]:
    """(ok, valor) — ok=False → usar defecto."""
    k = spec.kind
    if k in ("float", "int"):
        v = _num(raw)
        if v is None:
            return False, None
        v = _clamp(v, spec.lo, spec.hi)  # type: ignore[arg-type]
        return True, (int(round(v)) if k == "int" else v)
    if k == "bool":
        return (True, raw) if isinstance(raw, bool) else (False, None)
    if k == "enum":
        allowed = {c for c, _ in spec.choices}
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and str(int(raw)) in allowed:
            raw = str(int(raw))
        return (True, raw) if isinstance(raw, str) and raw in allowed else (False, None)
    if k == "curve":
        c = normalize_curve(raw)
        return (c is not None), c
    if k == "wb3":
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            vals = [_num(x) for x in raw]
            if all(v is not None for v in vals):
                return True, tuple(_clamp(v, spec.lo, spec.hi) for v in vals)  # type: ignore[arg-type]
        return False, None
    return False, None


# --------------------------------------------------------------------- DevParams
def _make_fields() -> dict[str, Any]:
    ann: dict[str, Any] = {}
    ns: dict[str, Any] = {}
    for p in SCHEMA:
        ann[p.key] = Any
        d = p.default
        ns[p.key] = field(default=d) if not isinstance(d, (list, dict)) else field(default_factory=lambda d=d: d)
    ns["__annotations__"] = ann
    return ns


class _DevParamsBase:
    # -------------------------------------------------------------- parse
    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "DevParams":
        p = cls()  # type: ignore[call-arg]
        if not isinstance(d, Mapping):
            return p  # type: ignore[return-value]
        for spec in SCHEMA:
            if spec.key in d:
                ok, v = _parse_value(spec, d[spec.key])
                if ok:
                    setattr(p, spec.key, v)
        return p  # type: ignore[return-value]

    @classmethod
    def coerce(cls, params: "DevParams | Mapping[str, Any] | None") -> "DevParams":
        if isinstance(params, _DevParamsBase):
            return cls.from_dict(params.to_dict(full=True))  # revalida (pudo mutarse)
        return cls.from_dict(params)

    # ---------------------------------------------------------- serialize
    def to_dict(self, *, full: bool = True) -> dict[str, Any]:
        """Objeto DEV JSON. ``full=False`` omite los valores por defecto (archivo más chico)."""
        out: dict[str, Any] = {"version": DEV_VERSION}
        for spec in SCHEMA:
            v = getattr(self, spec.key)
            if spec.kind == "float":
                v = float(v)
            elif spec.kind == "wb3":
                v = [float(x) for x in v]
            elif spec.kind == "curve":
                v = [[float(x), float(y)] for x, y in v]
            if not full and self._is_default(spec, v):
                continue
            out[spec.key] = v
        return out

    @staticmethod
    def _is_default(spec: ParamSpec, v: Any) -> bool:
        d = spec.default
        if spec.kind == "curve":
            return is_identity_curve(v)
        if spec.kind == "wb3":
            return list(v) == [float(x) for x in d]
        if spec.kind == "float":
            return abs(float(v) - float(d)) < 1e-9
        return v == d

    # ------------------------------------------------------------- queries
    def is_default(self, key: str) -> bool:
        return self._is_default(SPECS[key], getattr(self, key))

    def changed_keys(self) -> list[str]:
        return [s.key for s in SCHEMA if not self.is_default(s.key)]

    def group_is_default(self, group: str) -> bool:
        return all(self.is_default(s.key) for s in SCHEMA if s.group == group)

    def reset(self, keys: Iterable[str] | None = None) -> None:
        for k in (keys if keys is not None else SPECS):
            setattr(self, k, SPECS[k].default)

    def reset_group(self, group: str) -> None:
        self.reset(s.key for s in SCHEMA if s.group == group)

    def copy(self) -> "DevParams":
        return type(self).from_dict(self.to_dict())  # type: ignore[return-value]

    def subset(self, keys: Iterable[str]) -> dict[str, Any]:
        full = self.to_dict()
        return {k: full[k] for k in keys if k in full}


DevParams = dataclass(type("DevParams", (_DevParamsBase,), _make_fields()))  # type: ignore[misc]
DevParams.__doc__ = "Parámetros de revelado (DEV v2). Un atributo por cada entrada de SCHEMA."
DevParams.__module__ = __name__

assert {f.name for f in fields(DevParams)} == set(SPECS)
