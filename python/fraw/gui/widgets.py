"""Small custom widgets: reset-on-double-click slider rows and the RGB histogram."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget

__all__ = ["ResetSlider", "SliderRow", "Histogram", "histogram_rgb"]


class ResetSlider(QSlider):
    """QSlider that returns to its default value on double click."""

    reset = Signal()

    def __init__(self, default: int, parent: QWidget | None = None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.default = default

    def mouseDoubleClickEvent(self, e):  # noqa: N802 (Qt API)
        self.setValue(self.default)
        self.reset.emit()
        e.accept()


class SliderRow(QWidget):
    """Label + value + slider for one float DEV parameter.

    ``valueChanged(float)`` is emitted with the real (scaled) value.
    """

    valueChanged = Signal(float)

    def __init__(self, label: str, lo: float, hi: float, default: float, step: float = 0.01,
                 fmt: str = "{:+.2f}", parent: QWidget | None = None):
        super().__init__(parent)
        self.lo, self.hi, self.step, self.fmt = lo, hi, step, fmt
        self.default_value = default
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(4)
        self.name = QLabel(label)
        self.value_label = QLabel()
        self.value_label.setObjectName("sliderValue")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.name)
        top.addStretch(1)
        top.addWidget(self.value_label)
        lay.addLayout(top)
        self.slider = ResetSlider(self._to_int(default))
        self.slider.setRange(self._to_int(lo), self._to_int(hi))
        self.slider.setValue(self._to_int(default))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(max(1, int(round(0.25 / step))))
        self.slider.setToolTip("Doble click para restablecer")
        if lo < 0 < hi:
            self.slider.setObjectName("centered")
        lay.addWidget(self.slider)
        self.slider.valueChanged.connect(self._changed)
        self._update_label()

    def _to_int(self, v: float) -> int:
        return int(round(v / self.step))

    def value(self) -> float:
        return round(self.slider.value() * self.step, 6)

    def setValue(self, v: float) -> None:  # noqa: N802 (Qt naming)
        v = min(max(float(v), self.lo), self.hi)
        self.slider.setValue(self._to_int(v))
        self._update_label()

    def set_silently(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.setValue(v)
        self.slider.blockSignals(False)

    def reset(self) -> None:
        self.setValue(self.default_value)

    def _update_label(self) -> None:
        self.value_label.setText(self.fmt.format(self.value()))

    def _changed(self, _i: int) -> None:
        self._update_label()
        self.valueChanged.emit(self.value())


def histogram_rgb(img: np.ndarray, bins: int = 256, max_samples: int = 150_000) -> np.ndarray:
    """(3, bins) counts of an H×W×3 uint8 image (strided subsample for speed)."""
    flat = img.reshape(-1, 3)
    step = max(1, flat.shape[0] // max_samples)
    s = flat[::step]
    return np.stack([np.bincount(s[:, c], minlength=256)[:256] for c in range(3)]).astype(np.float32)


class Histogram(QWidget):
    """Live RGB histogram (additive overlay, sqrt-scaled)."""

    COLORS = ((235, 72, 77), (63, 185, 80), (76, 141, 255))

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._h: np.ndarray | None = None

    def set_data(self, h: np.ndarray | None) -> None:
        self._h = h
        self.update()

    def paintEvent(self, _e):  # noqa: N802 (Qt API)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1, 1, -1, -1)
        p.setPen(QPen(QColor("#2E2D33"), 1))
        p.setBrush(QColor("#0E0D12"))
        p.drawRoundedRect(r, 8, 8)
        p.setPen(QPen(QColor(46, 45, 51), 1, Qt.PenStyle.DotLine))
        for i in (1, 2, 3):
            x = r.left() + r.width() * i / 4
            p.drawLine(QPointF(x, r.top() + 4), QPointF(x, r.bottom() - 4))
        if self._h is None:
            return
        h = np.sqrt(self._h)
        top = float(h[:, 1:-1].max()) if h.shape[1] > 2 else float(h.max())
        if top <= 0:
            return
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        w, hh = r.width() - 8, r.height() - 8
        x0, y0 = r.left() + 4, r.bottom() - 4
        n = h.shape[1]
        for c, (cr, cg, cb) in enumerate(self.COLORS):
            path = QPainterPath(QPointF(x0, y0))
            for i in range(n):
                v = min(1.0, h[c, i] / top)
                path.lineTo(QPointF(x0 + w * i / (n - 1), y0 - hh * v))
            path.lineTo(QPointF(x0 + w, y0))
            path.closeSubpath()
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(cr, cg, cb, 150))
            p.drawPath(path)
        p.end()
