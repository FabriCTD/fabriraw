"""Histograma RGB + luminancia con triángulos de recorte clicables, y utilidades."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["Histogram", "histogram_rgb"]


def histogram_rgb(img: np.ndarray, bins: int = 256, max_samples: int = 150_000) -> np.ndarray:
    """(3, bins) counts of an H×W×3 uint8 image (strided subsample for speed)."""
    flat = img.reshape(-1, 3)
    step = max(1, flat.shape[0] // max_samples)
    s = flat[::step]
    return np.stack([np.bincount(s[:, c], minlength=256)[:256] for c in range(3)]).astype(np.float32)


class Histogram(QWidget):
    """RGB (aditivo) + luminancia, escala raíz. Triángulos arriba a izquierda/derecha:
    se encienden si hay recorte de sombras/luces; click alterna el aviso en la foto."""

    COLORS = ((235, 72, 77), (63, 185, 80), (76, 141, 255))
    clipToggled = Signal(str, bool)  # "shadows" | "highlights", on

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._h: np.ndarray | None = None
        self._l: np.ndarray | None = None
        self.clip_sh = 0.0
        self.clip_hl = 0.0
        self.show_sh = False
        self.show_hl = False
        self.info = ""
        self.setMouseTracking(True)
        self.setToolTip("Histograma · click en los triángulos: mostrar recorte de sombras / luces (J)")

    def set_data(self, h) -> None:
        if isinstance(h, dict):
            self._h = np.stack([np.asarray(h[c], dtype=np.float32)[:256] for c in "rgb"])
            self._l = np.asarray(h.get("l"), dtype=np.float32)[:256] if h.get("l") is not None else None
            self.clip_sh = float(h.get("clip_shadows", 0.0) or 0.0)
            self.clip_hl = float(h.get("clip_highlights", 0.0) or 0.0)
        else:
            self._h = h
            self._l = None
        self.update()

    def set_clip_state(self, sh: bool, hl: bool) -> None:
        self.show_sh, self.show_hl = sh, hl
        self.update()

    def _tri_rects(self) -> tuple[QRectF, QRectF]:
        r = self.rect().adjusted(1, 1, -1, -1)
        return QRectF(r.left() + 4, r.top() + 4, 14, 12), QRectF(r.right() - 18, r.top() + 4, 14, 12)

    def mousePressEvent(self, e):  # noqa: N802
        a, b = self._tri_rects()
        if a.adjusted(-4, -4, 4, 4).contains(e.position()):
            self.show_sh = not self.show_sh
            self.clipToggled.emit("shadows", self.show_sh)
            self.update()
        elif b.adjusted(-4, -4, 4, 4).contains(e.position()):
            self.show_hl = not self.show_hl
            self.clipToggled.emit("highlights", self.show_hl)
            self.update()

    def paintEvent(self, _e):  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor("#2E2D33"), 1))
        p.setBrush(QColor("#0E0D12"))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(QPen(QColor(46, 45, 51), 1, Qt.PenStyle.DotLine))
        for i in (1, 2, 3):
            x = r.left() + r.width() * i / 4
            p.drawLine(QPointF(x, r.top() + 4), QPointF(x, r.bottom() - 4))
        self._paint_triangles(p)
        if self._h is None:
            p.end()
            return
        h = np.sqrt(self._h)
        top = float(h[:, 1:-1].max()) if h.shape[1] > 2 else float(h.max())
        w, hh = r.width() - 8, r.height() - 22
        x0, y0 = r.left() + 4, r.bottom() - 4
        n = h.shape[1]
        if top > 0:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            for c, (cr, cg, cb) in enumerate(self.COLORS):
                path = QPainterPath(QPointF(x0, y0))
                for i in range(n):
                    v = min(1.0, h[c, i] / top)
                    path.lineTo(QPointF(x0 + w * i / (n - 1), y0 - hh * v))
                path.lineTo(QPointF(x0 + w, y0))
                path.closeSubpath()
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(cr, cg, cb, 140))
                p.drawPath(path)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            if self._l is not None:
                lv = np.sqrt(self._l)
                lt = float(lv[1:-1].max()) if lv.size > 2 else float(lv.max())
                if lt > 0:
                    path = QPainterPath(QPointF(x0, y0 - hh * min(1.0, lv[0] / lt)))
                    for i in range(1, len(lv)):
                        path.lineTo(QPointF(x0 + w * i / (len(lv) - 1), y0 - hh * min(1.0, lv[i] / lt)))
                    p.setPen(QPen(QColor(237, 235, 240, 190), 1.2))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(path)
        if self.info:
            p.setPen(QColor("#A6A3AD"))
            p.drawText(r.adjusted(24, 2, -24, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.info)
        p.end()

    def _paint_triangles(self, p: QPainter) -> None:
        a, b = self._tri_rects()
        for rect, frac, on, col, left in ((a, self.clip_sh, self.show_sh, QColor(64, 128, 255), True),
                                          (b, self.clip_hl, self.show_hl, QColor(255, 72, 72), False)):
            poly = QPolygonF([rect.topLeft(), rect.topRight(), rect.bottomLeft()] if left
                             else [rect.topLeft(), rect.topRight(), rect.bottomRight()])
            active = frac > 0.0005
            fill = QColor(col) if active else QColor("#45434C")
            if not on and active:
                fill.setAlpha(150)
            p.setPen(QPen(QColor("#EDEBF0") if on else QColor("#2E2D33"), 1))
            p.setBrush(fill)
            p.drawPolygon(poly)
