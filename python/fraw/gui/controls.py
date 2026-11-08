"""Controles de revelado generados a partir de :data:`fraw.devparams.SCHEMA`.

* :class:`ParamSlider` — deslizador propio (pintado a mano): relleno desde el
  centro en ajustes bipolares, fondo degradé (temperatura, matiz, HSL, tono),
  rueda (fino / Shift grueso / Ctrl muy fino), flechas, doble click = defecto,
  mapeo en *mired* para la temperatura en Kelvin.
* :class:`ValueField` — campo numérico editable: muestra ``valor × ui_scale`` con
  ``decimals`` y ``unit``; Enter aplica, Esc revierte, rueda y flechas ajustan.
* :class:`ParamRow` — etiqueta + deslizador + campo para un ``float``/``int``.
* :class:`EnumControl` (combo o botones segmentados) y :class:`ToggleSwitch`.

Todas las señales ``changed(value, final)`` llevan el valor *guardado* (sin
escalar) y ``final`` = False mientras dura una interacción continua (arrastre,
rueda, flechas) y True al terminarla; así la ventana renderiza en borrador y
agrupa los pasos del historial.
"""

from __future__ import annotations

import colorsys
import math
import re
import unicodedata
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from ..devparams import HSL_BAND_HUES, SCHEMA, ParamSpec

__all__ = [
    "ParamSlider", "ValueField", "ParamRow", "EnumControl", "ToggleSwitch", "ClickLabel",
    "make_control", "format_value", "parse_value", "display_name", "history_label", "fold",
    "COMMIT_MS",
]

COMMIT_MS = 400  # rueda/flechas: tras esta pausa la interacción se da por terminada
AMBER = QColor("#F0B232")
TRACK = QColor("#2E2D33")


# --------------------------------------------------------------------- text
def fold(s: str) -> str:
    """Minúsculas sin acentos (búsqueda)."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def format_value(spec: ParamSpec, v: float, *, unit: bool = True, sign: bool | None = None) -> str:
    """Valor guardado → texto visible (× ui_scale, coma decimal, unidad)."""
    d = v * spec.ui_scale
    dec = max(0, int(spec.decimals))
    if sign is None:
        sign = spec.lo is not None and spec.lo < 0 < (spec.hi or 0)
    txt = f"{d:+.{dec}f}" if sign and abs(d) >= 0.5 * 10**-dec else f"{d:.{dec}f}"
    if txt in ("-0", "+0") or re.fullmatch(r"[+-]0[.,]0+", txt):
        txt = txt[1:]
    txt = txt.replace(".", ",")
    if unit and spec.unit:
        txt += ("" if spec.unit in ("°", "%") else " ") + spec.unit
    return txt


def parse_value(spec: ParamSpec, text: str) -> float | None:
    """Texto escrito por el usuario (en unidades visibles) → valor guardado recortado."""
    t = text.strip().replace("−", "-")
    if spec.unit:
        t = t.replace(spec.unit, "")
    t = t.replace(" ", "").replace(",", ".")
    if not t:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    v /= spec.ui_scale or 1.0
    lo, hi = float(spec.lo), float(spec.hi)  # type: ignore[arg-type]
    return min(hi, max(lo, v))


_DUP_LABELS = {lbl for lbl in (s.label for s in SCHEMA) if sum(1 for s in SCHEMA if s.label == lbl) > 1}


def display_name(spec: ParamSpec) -> str:
    """Nombre sin ambigüedad (p. ej. «Saturación · Rojo» en HSL)."""
    if spec.label in _DUP_LABELS and spec.section and spec.section != spec.label:
        if spec.group == "hsl":
            return f"{spec.section} {spec.label.lower()}"
        return f"{spec.section} · {spec.label.lower()}"
    return spec.label


def history_label(spec: ParamSpec, value: Any) -> str:
    if spec.kind in ("float", "int"):
        return f"{display_name(spec)} {format_value(spec, float(value), unit=False)}"
    if spec.kind == "bool":
        return f"{display_name(spec)}: {'sí' if value else 'no'}"
    if spec.kind == "enum":
        lbl = dict(spec.choices).get(str(value), str(value))
        return f"{display_name(spec)}: {lbl}"
    return display_name(spec)


# --------------------------------------------------------------- gradients
def _hsv(h: float, s: float = 0.75, v: float = 0.85) -> QColor:
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, s, v)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


def gradient_for(spec: ParamSpec) -> list[tuple[float, QColor]] | None:
    k = spec.key
    if k in ("wb_kelvin", "temperature_shift"):
        return [(0.0, QColor(70, 120, 230)), (0.5, QColor(200, 200, 200)), (1.0, QColor(235, 200, 60))]
    if k in ("wb_kelvin_tint", "tint_shift", "cal_shadows_tint"):
        return [(0.0, QColor(70, 190, 90)), (0.5, QColor(200, 200, 200)), (1.0, QColor(215, 80, 200))]
    m = re.fullmatch(r"hsl_(\w+)_([hsl])", k)
    if m and m.group(1) in HSL_BAND_HUES:
        h0 = HSL_BAND_HUES[m.group(1)]
        if m.group(2) == "h":
            return [(0.0, _hsv(h0 - 30)), (0.5, _hsv(h0)), (1.0, _hsv(h0 + 30))]
        if m.group(2) == "s":
            return [(0.0, QColor(128, 128, 128)), (1.0, _hsv(h0, 1.0, 0.95))]
        return [(0.0, _hsv(h0, 0.8, 0.25)), (1.0, _hsv(h0, 0.35, 1.0))]
    if re.fullmatch(r"grade_\w+_hue", k):
        return [(i / 6.0, _hsv(60 * i)) for i in range(7)]
    m = re.fullmatch(r"cal_(red|green|blue)_hue", k)
    if m:
        h0 = {"red": 0, "green": 120, "blue": 240}[m.group(1)]
        return [(0.0, _hsv(h0 - 40)), (0.5, _hsv(h0)), (1.0, _hsv(h0 + 40))]
    m = re.fullmatch(r"cal_(red|green|blue)_sat", k)
    if m:
        h0 = {"red": 0, "green": 120, "blue": 240}[m.group(1)]
        return [(0.0, QColor(128, 128, 128)), (1.0, _hsv(h0, 1.0, 0.95))]
    return None


# ---------------------------------------------------------------- ClickLabel
class ClickLabel(QLabel):
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        self.doubleClicked.emit()
        e.accept()


# --------------------------------------------------------------- ParamSlider
class ParamSlider(QWidget):
    """Deslizador float con relleno bipolar y degradés. ``changed(value, final)``."""

    changed = Signal(float, bool)

    def __init__(self, spec: ParamSpec, parent: QWidget | None = None, *, compact: bool = False):
        super().__init__(parent)
        self.spec = spec
        self.lo, self.hi = float(spec.lo), float(spec.hi)  # type: ignore[arg-type]
        self.step = float(spec.step) or (self.hi - self.lo) / 200.0
        self.default = float(spec.default)
        self._value = self.default
        self.down = False
        self.mired = spec.key == "wb_kelvin"
        mid = 0.5 * (self.lo + self.hi)
        self.origin = self.default if abs(mid - self.default) < 1e-9 else self.lo
        self.gradient = gradient_for(spec)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(18 if compact else 22)
        self.setMinimumWidth(60)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._commit = QTimer(self)
        self._commit.setSingleShot(True)
        self._commit.setInterval(COMMIT_MS)
        self._commit.timeout.connect(lambda: self.changed.emit(self._value, True))
        if spec.tooltip:
            self.setToolTip(spec.tooltip + "\nDoble click: restablecer")
        else:
            self.setToolTip("Doble click: restablecer · Rueda: ajustar (Shift grueso, Ctrl fino)")

    # ------------------------------------------------------------- mapping
    def _to_pos(self, v: float) -> float:
        if self.mired:
            a, b = 1e6 / self.lo, 1e6 / self.hi
            return (a - 1e6 / max(v, 1e-6)) / (a - b)
        return (v - self.lo) / (self.hi - self.lo)

    def _from_pos(self, t: float) -> float:
        t = min(1.0, max(0.0, t))
        if self.mired:
            a, b = 1e6 / self.lo, 1e6 / self.hi
            return 1e6 / (a - t * (a - b))
        return self.lo + t * (self.hi - self.lo)

    def _snap(self, v: float) -> float:
        v = min(self.hi, max(self.lo, v))
        q = self.step / 10.0
        return round(round((v - self.lo) / q) * q + self.lo, 9)

    def _track(self) -> QRectF:
        m = 7.0
        return QRectF(m, self.height() / 2 - 2, self.width() - 2 * m, 4)

    # --------------------------------------------------------------- value
    def value(self) -> float:
        return self._value

    def setValue(self, v: float, *, emit: bool = True, final: bool = True) -> None:  # noqa: N802
        v = self._snap(float(v))
        changed = abs(v - self._value) > 1e-12
        self._value = v
        self.update()
        if emit and (changed or final):
            self.changed.emit(v, final)

    def set_silently(self, v: float) -> None:
        self._value = min(self.hi, max(self.lo, float(v)))
        self.update()

    def reset(self) -> None:
        self._commit.stop()
        self.setValue(self.default, final=True)

    def nudge(self, steps: float) -> None:
        if self.mired:  # paso proporcional en Kelvin (≈ constante en mired)
            v = self._value * (1.0 + 0.01 * steps * (self.step / 50.0))
        else:
            v = self._value + steps * self.step
        self.setValue(v, final=False)
        self._commit.start()

    # -------------------------------------------------------------- events
    def _value_at(self, x: float) -> float:
        tr = self._track()
        return self._from_pos((x - tr.left()) / max(1.0, tr.width()))

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self.down = True
        self.setFocus()
        self.setValue(self._value_at(e.position().x()), final=False)

    def mouseMoveEvent(self, e):  # noqa: N802
        if self.down:
            fine = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            v = self._value_at(e.position().x())
            if fine:  # Shift durante el arrastre: 1/4 de velocidad
                v = self._value + (v - self._value) * 0.25
            self.setValue(v, final=False)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self.down and e.button() == Qt.MouseButton.LeftButton:
            self.down = False
            self.changed.emit(self._value, True)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        self.down = False
        self.reset()
        e.accept()

    @staticmethod
    def _mult(mods) -> float:
        if mods & Qt.KeyboardModifier.ControlModifier:
            return 0.1
        if mods & Qt.KeyboardModifier.ShiftModifier:
            return 10.0
        return 1.0

    def wheelEvent(self, e):  # noqa: N802
        d = e.angleDelta()
        n = (d.y() or d.x()) / 120.0
        if n:
            self.nudge(n * self._mult(e.modifiers()))
        e.accept()

    def keyPressEvent(self, e):  # noqa: N802
        k = e.key()
        m = self._mult(e.modifiers())
        if k in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.nudge(m)
        elif k in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.nudge(-m)
        elif k == Qt.Key.Key_PageUp:
            self.nudge(10 * m)
        elif k == Qt.Key.Key_PageDown:
            self.nudge(-10 * m)
        elif k in (Qt.Key.Key_Home, Qt.Key.Key_End):
            self.setValue(self.lo if k == Qt.Key.Key_Home else self.hi, final=True)
        elif k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.reset()
        else:
            super().keyPressEvent(e)

    # --------------------------------------------------------------- paint
    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(160, self.height())

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        en = self.isEnabled()
        tr = self._track()
        if self.gradient:
            g = QLinearGradient(tr.left(), 0, tr.right(), 0)
            for pos, c in self.gradient:
                cc = QColor(c)
                if not en:
                    cc.setAlpha(70)
                g.setColorAt(pos, cc)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(g)
            p.drawRoundedRect(tr.adjusted(0, -0.5, 0, 0.5), 2.5, 2.5)
        else:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(TRACK)
            p.drawRoundedRect(tr, 2, 2)
            x0 = tr.left() + tr.width() * self._to_pos(self.origin)
            x1 = tr.left() + tr.width() * self._to_pos(self._value)
            c = QColor(AMBER) if en else QColor("#5E5B66")
            p.setBrush(c)
            p.drawRoundedRect(QRectF(min(x0, x1), tr.top(), abs(x1 - x0), tr.height()), 2, 2)
        if self.origin != self.lo:  # marca del centro
            xc = tr.left() + tr.width() * self._to_pos(self.origin)
            p.setPen(QPen(QColor("#6A6772"), 1))
            p.drawLine(QPointF(xc, tr.top() - 4), QPointF(xc, tr.bottom() + 4))
        x = tr.left() + tr.width() * self._to_pos(self._value)
        r = 6.0 if (self.down or self.hasFocus()) else 5.0
        p.setPen(QPen(QColor(AMBER) if en else QColor("#5E5B66"), 2))
        p.setBrush(QColor("#EDEBF0") if en else QColor("#45434C"))
        p.drawEllipse(QPointF(x, self.height() / 2), r, r)
        p.end()


# ---------------------------------------------------------------- ValueField
class ValueField(QLineEdit):
    """Campo numérico: escribe en unidades visibles; Enter aplica; Esc revierte."""

    committed = Signal(float, bool)

    def __init__(self, spec: ParamSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec = spec
        self._value = float(spec.default)
        self.setObjectName("valueField")
        self.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.setFixedWidth(76 if spec.unit else 54)
        self.setToolTip(f"{display_name(spec)}: escribí un valor y Enter · rueda/flechas ajustan")
        self._commit = QTimer(self)
        self._commit.setSingleShot(True)
        self._commit.setInterval(COMMIT_MS)
        self._commit.timeout.connect(lambda: self.committed.emit(self._value, True))
        self.returnPressed.connect(self.apply_text)
        self.editingFinished.connect(self._on_finished)
        self.show_value(self._value)

    def show_value(self, v: float) -> None:
        self._value = float(v)
        if not self.hasFocus() or not self.isModified():
            self.setText(format_value(self.spec, self._value))
            self.setModified(False)

    def apply_text(self) -> bool:
        v = parse_value(self.spec, self.text())
        if v is None:
            self.setStyleSheet("border-color: #E5484D;")
            QTimer.singleShot(600, lambda: self.setStyleSheet(""))
            self.setText(format_value(self.spec, self._value))
            self.setModified(False)
            return False
        self._value = v
        self.setModified(False)
        self.setText(format_value(self.spec, v))
        self.committed.emit(v, True)
        return True

    def _on_finished(self) -> None:
        if self.isModified():
            self.apply_text()

    def _nudge(self, n: float) -> None:
        v = min(float(self.spec.hi), max(float(self.spec.lo), self._value + n * float(self.spec.step)))  # type: ignore[arg-type]
        self._value = round(v, 9)
        self.setText(format_value(self.spec, self._value))
        self.setModified(False)
        self.committed.emit(self._value, False)
        self._commit.start()

    def keyPressEvent(self, e):  # noqa: N802
        k = e.key()
        m = ParamSlider._mult(e.modifiers())
        if k == Qt.Key.Key_Up:
            self._nudge(m)
        elif k == Qt.Key.Key_Down:
            self._nudge(-m)
        elif k == Qt.Key.Key_Escape:
            self.setText(format_value(self.spec, self._value))
            self.setModified(False)
            self.clearFocus()
        else:
            super().keyPressEvent(e)

    def wheelEvent(self, e):  # noqa: N802
        d = e.angleDelta()
        n = (d.y() or d.x()) / 120.0
        if n:
            self._nudge(n * ParamSlider._mult(e.modifiers()))
        e.accept()

    def focusInEvent(self, e):  # noqa: N802
        super().focusInEvent(e)
        QTimer.singleShot(0, self.selectAll)


# ------------------------------------------------------------------ ParamRow
class ParamRow(QWidget):
    """Etiqueta + deslizador + campo numérico para un ajuste float/int.

    ``changed(key, value, final)``. API compatible con el antiguo ``SliderRow``
    (``setValue``, ``value``, ``set_silently``, ``reset``, atributo ``slider``).
    """

    changed = Signal(str, object, bool)

    def __init__(self, spec: ParamSpec, parent: QWidget | None = None, *, label: str | None = None):
        super().__init__(parent)
        self.spec = spec
        self.key = spec.key
        self.default_value = float(spec.default)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(6)
        self.name = ClickLabel(label if label is not None else spec.label)
        self.name.setObjectName("paramLabel")
        self.name.setFixedWidth(122)
        self.name.setToolTip((spec.tooltip + "\n" if spec.tooltip else "") + "Doble click: restablecer")
        self.slider = ParamSlider(spec)
        self.field = ValueField(spec)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.field)
        self.name.doubleClicked.connect(self.reset)
        self.slider.changed.connect(self._from_slider)
        self.field.committed.connect(self._from_field)
        self._sync_label()

    def _sync_label(self) -> None:
        self.mark_changed(abs(self.slider.value() - self.default_value) > 1e-9)

    def mark_changed(self, changed: bool) -> None:
        if self.name.property("changed") != changed:
            self.name.setProperty("changed", changed)
            self.name.style().unpolish(self.name)
            self.name.style().polish(self.name)

    def _from_slider(self, v: float, final: bool) -> None:
        self.field.show_value(v)
        self._sync_label()
        self.changed.emit(self.key, v, final)

    def _from_field(self, v: float, final: bool) -> None:
        self.slider.set_silently(v)
        self._sync_label()
        self.changed.emit(self.key, self.slider.value() if abs(self.slider.value() - v) < 1e-12 else v, final)

    # API
    def value(self) -> float:
        return self.slider.value()

    def setValue(self, v: float) -> None:  # noqa: N802
        self.slider.setValue(v, final=True)

    def set_silently(self, v: Any) -> None:
        self.slider.set_silently(float(v))
        self.field.show_value(float(v))
        self._sync_label()

    def reset(self) -> None:
        self.slider.reset()


# ---------------------------------------------------------------- ToggleSwitch
class ToggleSwitch(QAbstractButton):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(34, 18)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(34, 18)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        on = self.isChecked()
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        p.setPen(QPen(QColor("#F0B232") if on else QColor("#45434C"), 1))
        p.setBrush(QColor("#3A3320") if on else QColor("#1A1920"))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 4
        x = r.right() - d - 2 if on else r.left() + 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#F0B232") if on else QColor("#A6A3AD"))
        p.drawEllipse(QRectF(x, r.top() + 2, d, d))
        p.end()


class BoolRow(QWidget):
    changed = Signal(str, object, bool)

    def __init__(self, spec: ParamSpec, parent: QWidget | None = None):
        super().__init__(parent)
        self.spec, self.key = spec, spec.key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.name = ClickLabel(spec.label)
        self.name.setObjectName("paramLabel")
        self.toggle = ToggleSwitch()
        lay.addWidget(self.name, 1)
        lay.addWidget(self.toggle)
        self.toggle.toggled.connect(lambda on: self.changed.emit(self.key, bool(on), True))
        self.name.doubleClicked.connect(lambda: self.setValue(bool(spec.default)))

    def value(self) -> bool:
        return self.toggle.isChecked()

    def setValue(self, v: bool) -> None:  # noqa: N802
        self.toggle.setChecked(bool(v))

    def set_silently(self, v: Any) -> None:
        self.toggle.blockSignals(True)
        self.toggle.setChecked(bool(v))
        self.toggle.blockSignals(False)
        self.toggle.update()

    def reset(self) -> None:
        self.setValue(bool(self.spec.default))


# ---------------------------------------------------------------- EnumControl
class EnumControl(QWidget):
    """Combo (muchas opciones) o botones segmentados (≤ 3 opciones cortas)."""

    changed = Signal(str, object, bool)

    def __init__(self, spec: ParamSpec, parent: QWidget | None = None, *, segmented: bool | None = None):
        super().__init__(parent)
        self.spec, self.key = spec, spec.key
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)
        self.name = ClickLabel(spec.label)
        self.name.setObjectName("paramLabel")
        self.name.setFixedWidth(122)
        self.name.doubleClicked.connect(self.reset)
        if spec.tooltip:
            self.setToolTip(spec.tooltip)
        lay.addWidget(self.name)
        if segmented is None:
            segmented = len(spec.choices) <= 3 and sum(len(lbl) for _, lbl in spec.choices) <= 30
        self.segmented = segmented
        self.combo: QComboBox | None = None
        self.buttons: dict[str, QToolButton] = {}
        if segmented:
            box = QWidget()
            bl = QHBoxLayout(box)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(0)
            self.group = QButtonGroup(self)
            self.group.setExclusive(True)
            for i, (val, lbl) in enumerate(spec.choices):
                b = QToolButton()
                b.setText(lbl)
                b.setCheckable(True)
                b.setObjectName("segFirst" if i == 0 else "segLast" if i == len(spec.choices) - 1 else "segMid")
                b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                self.group.addButton(b)
                self.buttons[val] = b
                bl.addWidget(b)
                b.clicked.connect(lambda _c=False, v=val: self.changed.emit(self.key, v, True))
            lay.addWidget(box, 1)
        else:
            self.combo = QComboBox()
            for val, lbl in spec.choices:
                self.combo.addItem(lbl, val)
            self.combo.currentIndexChanged.connect(lambda _i: self.changed.emit(self.key, self.combo.currentData(), True))
            lay.addWidget(self.combo, 1)
        self.set_silently(spec.default)

    def value(self) -> str:
        if self.combo is not None:
            return self.combo.currentData()
        for v, b in self.buttons.items():
            if b.isChecked():
                return v
        return self.spec.default

    def currentData(self) -> str:  # noqa: N802 — compatibilidad con QComboBox
        return self.value()

    def setValue(self, v: str) -> None:  # noqa: N802
        if self.combo is not None:
            i = self.combo.findData(str(v))
            if i >= 0:
                self.combo.setCurrentIndex(i)
        elif str(v) in self.buttons:
            self.buttons[str(v)].setChecked(True)
            self.changed.emit(self.key, str(v), True)

    def set_silently(self, v: Any) -> None:
        if self.combo is not None:
            self.combo.blockSignals(True)
            i = self.combo.findData(str(v))
            self.combo.setCurrentIndex(max(0, i))
            self.combo.blockSignals(False)
        elif str(v) in self.buttons:
            self.buttons[str(v)].setChecked(True)

    def reset(self) -> None:
        self.setValue(self.spec.default)


def make_control(spec: ParamSpec) -> QWidget | None:
    """Control para un ParamSpec no oculto (las curvas tienen editor propio)."""
    if spec.kind in ("float", "int"):
        return ParamRow(spec)
    if spec.kind == "bool":
        return BoolRow(spec)
    if spec.kind == "enum":
        return EnumControl(spec)
    return None

