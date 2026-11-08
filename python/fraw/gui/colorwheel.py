"""Rueda de color (tono = ángulo, saturación = radio) con punto arrastrable.

Ángulo 0° = rojo a la derecha, creciendo en sentido antihorario.
``changed(hue_deg, sat_0_1, final)``. Doble click vuelve al centro (sat 0).
Shift mientras se arrastra sólo cambia la saturación (mantiene el tono).
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["ColorWheel", "hue_sat_from_xy", "xy_from_hue_sat", "wheel_rgb"]


def hue_sat_from_xy(dx: float, dy: float, radius: float) -> tuple[float, float]:
    """Desplazamiento desde el centro (y hacia abajo positivo) → (tono °, saturación)."""
    sat = min(1.0, math.hypot(dx, dy) / max(1e-9, radius))
    hue = math.degrees(math.atan2(-dy, dx)) % 360.0
    return hue, sat


def xy_from_hue_sat(hue: float, sat: float, radius: float) -> tuple[float, float]:
    a = math.radians(hue)
    return math.cos(a) * sat * radius, -math.sin(a) * sat * radius


def wheel_rgb(size: int) -> np.ndarray:
    """Disco RGBA (size×size×4 uint8): tono por ángulo, saturación por radio."""
    c = (size - 1) / 2.0
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = x - c, y - c
    r = np.hypot(dx, dy) / c
    h = (np.degrees(np.arctan2(-dy, dx)) % 360.0) / 60.0
    s = np.clip(r, 0, 1) * 0.85
    v = np.full_like(r, 0.82)
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    rgb = np.zeros((size, size, 3), np.float32)
    for k, (a, b, cc) in enumerate(((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))):
        m = i == k
        rgb[m, 0], rgb[m, 1], rgb[m, 2] = a[m], b[m], cc[m]
    alpha = np.clip((1.0 - r) * c + 0.5, 0, 1)  # borde antialias
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = (rgb * 255).astype(np.uint8)
    out[..., 3] = (alpha * 255).astype(np.uint8)
    return out


class ColorWheel(QWidget):
    changed = Signal(float, float, bool)

    def __init__(self, parent: QWidget | None = None, size: int = 132):
        super().__init__(parent)
        self.hue = 0.0
        self.sat = 0.0
        self._down = False
        self._img: QImage | None = None
        self._buf: np.ndarray | None = None
        self.setMinimumSize(size, size)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(size)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Arrastrá el punto: ángulo = tono, distancia = saturación\nShift: sólo saturación · doble click: neutro")

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(self.height(), self.height())

    def radius(self) -> float:
        return min(self.width(), self.height()) / 2.0 - 6

    def center(self) -> QPointF:
        return QPointF(self.width() / 2.0, self.height() / 2.0)

    def set_value(self, hue: float, sat: float) -> None:
        self.hue, self.sat = float(hue) % 360.0, min(1.0, max(0.0, float(sat)))
        self.update()

    def _apply(self, pos: QPointF, keep_hue: bool, final: bool) -> None:
        c = self.center()
        hue, sat = hue_sat_from_xy(pos.x() - c.x(), pos.y() - c.y(), self.radius())
        if keep_hue and self.sat > 0:
            hue = self.hue
        self.hue, self.sat = round(hue, 1), round(sat, 3)
        self.update()
        self.changed.emit(self.hue, self.sat, final)

    def mousePressEvent(self, e):  # noqa: N802
        if e.button() == Qt.MouseButton.LeftButton:
            self._down = True
            self._apply(e.position(), bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier), False)

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._down:
            self._apply(e.position(), bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier), False)

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._down:
            self._down = False
            self.changed.emit(self.hue, self.sat, True)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        self._down = False
        self.hue, self.sat = 0.0, 0.0
        self.update()
        self.changed.emit(0.0, 0.0, True)
        e.accept()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.radius()
        d = int(2 * r)
        if self._img is None or self._img.width() != d:
            self._buf = wheel_rgb(max(8, d))
            self._img = QImage(self._buf.data, d, d, d * 4, QImage.Format.Format_RGBA8888)
        c = self.center()
        if not self.isEnabled():
            p.setOpacity(0.35)
        p.drawImage(QRectF(c.x() - r, c.y() - r, d, d), self._img)
        p.setPen(QPen(QColor("#2E2D33"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(c, r, r)
        p.setPen(QPen(QColor(0, 0, 0, 70), 1))
        p.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
        dx, dy = xy_from_hue_sat(self.hue, self.sat, r)
        q = QPointF(c.x() + dx, c.y() + dy)
        p.setPen(QPen(QColor("#09080C"), 3))
        p.drawEllipse(q, 5.5, 5.5)
        p.setPen(QPen(QColor("#F0B232") if self.sat > 0 else QColor("#EDEBF0"), 2))
        p.drawEllipse(q, 5.5, 5.5)
        p.end()
