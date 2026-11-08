"""Editor de curva de tonos por puntos (RGB / Rojo / Verde / Azul).

Funciones puras (sin Qt, testeables): :func:`add_point`, :func:`move_point`,
:func:`delete_point`, :func:`eval_curve` (interpolación cúbica monótona,
Fritsch–Carlson). Reglas: 2–16 puntos, x estrictamente creciente (separación
mínima ``MIN_DX``), coordenadas en [0, 1]; los extremos no se borran.

:class:`CurveEditor` — pestañas de canal, histograma de fondo, grilla 4×4,
click agrega, arrastrar mueve, doble click o arrastrar afuera borra, campos
Entrada/Salida (0–255) del punto seleccionado. Emite
``changed(key, [[x, y], …], final)`` con ``key`` en ``curve_master|red|green|blue``.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSpinBox, QTabBar, QToolButton, QVBoxLayout, QWidget

from ..devparams import CURVE_KEYS, CURVE_MAX_POINTS

__all__ = ["CurveEditor", "CurveCanvas", "add_point", "move_point", "delete_point", "eval_curve", "MIN_DX"]

MIN_DX = 0.01
CHANNELS = (("curve_master", "RGB", QColor("#EDEBF0")), ("curve_red", "Rojo", QColor("#EB484D")),
            ("curve_green", "Verde", QColor("#3FB950")), ("curve_blue", "Azul", QColor("#4C8DFF")))
HIST_KEY = {"curve_master": "l", "curve_red": "r", "curve_green": "g", "curve_blue": "b"}

Pts = list[list[float]]


def _clean(points) -> Pts:
    return [[float(min(1.0, max(0.0, x))), float(min(1.0, max(0.0, y)))] for x, y in points]


def add_point(points, x: float, y: float) -> tuple[Pts, int]:
    """Agrega (x, y). Devuelve (puntos, índice) — índice −1 si no se pudo (lleno o muy cerca)."""
    pts = _clean(points)
    x, y = float(min(1.0, max(0.0, x))), float(min(1.0, max(0.0, y)))
    if len(pts) >= CURVE_MAX_POINTS or any(abs(px - x) < MIN_DX for px, _ in pts):
        return pts, -1
    pts.append([x, y])
    pts.sort(key=lambda p: p[0])
    return pts, next(i for i, p in enumerate(pts) if p[0] == x and p[1] == y)


def move_point(points, i: int, x: float, y: float) -> Pts:
    """Mueve el punto ``i`` manteniendo x estrictamente creciente."""
    pts = _clean(points)
    lo = pts[i - 1][0] + MIN_DX if i > 0 else 0.0
    hi = pts[i + 1][0] - MIN_DX if i < len(pts) - 1 else 1.0
    pts[i] = [float(min(hi, max(lo, x))), float(min(1.0, max(0.0, y)))]
    return pts


def delete_point(points, i: int) -> Pts:
    """Borra el punto ``i`` (no los extremos ni si quedarían < 2)."""
    pts = _clean(points)
    if len(pts) <= 2 or i <= 0 or i >= len(pts) - 1:
        return pts
    del pts[i]
    return pts


def eval_curve(points, xs: np.ndarray) -> np.ndarray:
    """Interpolación cúbica de Hermite monótona (Fritsch–Carlson); constante fuera de rango."""
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    xs = np.asarray(xs, dtype=np.float64)
    n = len(x)
    if n == 2:
        t = np.clip((xs - x[0]) / max(1e-9, x[1] - x[0]), 0, 1)
        return y[0] + t * (y[1] - y[0])
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    xc = np.clip(xs, x[0], x[-1])
    k = np.clip(np.searchsorted(x, xc) - 1, 0, n - 2)
    t = (xc - x[k]) / h[k]
    t2, t3 = t * t, t * t * t
    out = ((2 * t3 - 3 * t2 + 1) * y[k] + (t3 - 2 * t2 + t) * h[k] * m[k]
           + (-2 * t3 + 3 * t2) * y[k + 1] + (t3 - t2) * h[k] * m[k + 1])
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------- widget
class CurveCanvas(QWidget):
    """Área de dibujo/edición de una curva."""

    edited = Signal(object, bool)  # puntos, final
    selectionChanged = Signal(int)

    PAD = 8
    HIT = 9

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.points: Pts = [[0.0, 0.0], [1.0, 1.0]]
        self.others: list[tuple[Pts, QColor]] = []
        self.color = QColor("#EDEBF0")
        self.hist: np.ndarray | None = None
        self.selected = -1
        self._drag = -1
        self._outside = False
        self._hover: QPointF | None = None

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return w

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    # geometry
    def area(self) -> QRectF:
        s = min(self.width(), self.height()) - 2 * self.PAD
        return QRectF((self.width() - s) / 2, self.PAD, s, s)

    def to_px(self, x: float, y: float) -> QPointF:
        a = self.area()
        return QPointF(a.left() + x * a.width(), a.bottom() - y * a.height())

    def to_curve(self, p: QPointF) -> tuple[float, float]:
        a = self.area()
        return (p.x() - a.left()) / a.width(), (a.bottom() - p.y()) / a.height()

    def hit(self, p: QPointF) -> int:
        best, bd = -1, float(self.HIT)
        for i, (x, y) in enumerate(self.points):
            q = self.to_px(x, y)
            d = ((q.x() - p.x()) ** 2 + (q.y() - p.y()) ** 2) ** 0.5
            if d <= bd:
                best, bd = i, d
        return best

    def set_points(self, pts, keep_selection: bool = True) -> None:
        self.points = _clean(pts)
        if not keep_selection or self.selected >= len(self.points):
            self.selected = -1
        self.update()

    # events
    def mousePressEvent(self, e):  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton:
            return
        p = e.position()
        i = self.hit(p)
        if i < 0:
            x, y = self.to_curve(p)
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return
            # el punto nuevo nace sobre la curva (no salta la imagen)
            ycur = float(eval_curve(self.points, np.array([x]))[0])
            if abs(ycur - y) * self.area().height() < 14:
                y = ycur
            pts, i = add_point(self.points, x, y)
            if i < 0:
                return
            self.points = pts
            self.edited.emit(self.points, False)
        self.selected = i
        self._drag = i
        self._outside = False
        self.selectionChanged.emit(i)
        self.update()

    def mouseMoveEvent(self, e):  # noqa: N802
        self._hover = e.position()
        if self._drag >= 0:
            x, y = self.to_curve(e.position())
            a = self.area()
            far = 24 / a.width()
            interior = 0 < self._drag < len(self.points) - 1
            self._outside = interior and len(self.points) > 2 and (x < -far or x > 1 + far or y < -far or y > 1 + far)
            self.points = move_point(self.points, self._drag, x, y)
            self.edited.emit(self.points, False)
            self.selectionChanged.emit(self._drag)
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._drag >= 0:
            if self._outside:
                self.points = delete_point(self.points, self._drag)
                self.selected = -1
                self.selectionChanged.emit(-1)
            self._drag = -1
            self._outside = False
            self.edited.emit(self.points, True)
            self.update()

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        i = self.hit(e.position())
        if i > 0 and i < len(self.points) - 1:
            self.points = delete_point(self.points, i)
            self.selected = -1
            self._drag = -1
            self.selectionChanged.emit(-1)
            self.edited.emit(self.points, True)
            self.update()
        e.accept()

    def leaveEvent(self, _e):  # noqa: N802
        self._hover = None
        self.update()

    def keyPressEvent(self, e):  # noqa: N802
        i = self.selected
        if i < 0:
            return super().keyPressEvent(e)
        step = (10 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1) / 255.0
        x, y = self.points[i]
        k = e.key()
        if k in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.points = delete_point(self.points, i)
            self.selected = -1
            self.selectionChanged.emit(-1)
            self.edited.emit(self.points, True)
        elif k in (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right):
            dx = step if k == Qt.Key.Key_Right else -step if k == Qt.Key.Key_Left else 0
            dy = step if k == Qt.Key.Key_Up else -step if k == Qt.Key.Key_Down else 0
            self.points = move_point(self.points, i, x + dx, y + dy)
            self.selectionChanged.emit(i)
            self.edited.emit(self.points, True)
        else:
            return super().keyPressEvent(e)
        self.update()

    # paint
    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        a = self.area()
        p.setPen(QPen(QColor("#2E2D33"), 1))
        p.setBrush(QColor("#0E0D12"))
        p.drawRoundedRect(a.adjusted(-1, -1, 1, 1), 4, 4)
        if self.hist is not None and self.hist.max() > 0:
            h = np.sqrt(self.hist.astype(np.float64))
            top = float(h[1:-1].max()) if len(h) > 2 else float(h.max())
            if top > 0:
                path = QPainterPath(QPointF(a.left(), a.bottom()))
                n = len(h)
                for i in range(n):
                    path.lineTo(QPointF(a.left() + a.width() * i / (n - 1), a.bottom() - a.height() * 0.9 * min(1.0, h[i] / top)))
                path.lineTo(QPointF(a.right(), a.bottom()))
                path.closeSubpath()
                c = QColor(self.color)
                c.setAlpha(38)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(c)
                p.drawPath(path)
        p.setPen(QPen(QColor(46, 45, 51), 1))
        for i in (1, 2, 3):
            p.drawLine(QPointF(a.left() + a.width() * i / 4, a.top()), QPointF(a.left() + a.width() * i / 4, a.bottom()))
            p.drawLine(QPointF(a.left(), a.top() + a.height() * i / 4), QPointF(a.right(), a.top() + a.height() * i / 4))
        p.setPen(QPen(QColor("#45434C"), 1, Qt.PenStyle.DashLine))
        p.drawLine(a.bottomLeft(), a.topRight())
        xs = np.linspace(0, 1, 96)
        for pts, col in self.others:
            c = QColor(col)
            c.setAlpha(90)
            self._curve(p, pts, xs, QPen(c, 1))
        self._curve(p, self.points, xs, QPen(self.color, 2))
        for i, (x, y) in enumerate(self.points):
            q = self.to_px(x, y)
            sel = i == self.selected
            p.setPen(QPen(QColor("#F0B232") if sel else self.color, 2))
            p.setBrush(QColor("#F0B232") if sel else QColor("#09080C"))
            if i == self._drag and self._outside:
                p.setPen(QPen(QColor("#E5484D"), 2))
            p.drawEllipse(q, 4.5, 4.5)
        if self._hover is not None and self._drag < 0 and a.contains(self._hover):
            x, _ = self.to_curve(self._hover)
            y = float(eval_curve(self.points, np.array([x]))[0])
            p.setPen(QColor("#A6A3AD"))
            p.drawText(a.adjusted(6, 4, -6, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft,
                       f"{x * 255:.0f} → {y * 255:.0f}")
        p.end()

    def _curve(self, p: QPainter, pts, xs: np.ndarray, pen: QPen) -> None:
        ys = eval_curve(pts, xs)
        path = QPainterPath(self.to_px(xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            path.lineTo(self.to_px(x, y))
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)


class CurveEditor(QWidget):
    """Pestañas RGB/Rojo/Verde/Azul + lienzo + Entrada/Salida del punto seleccionado."""

    changed = Signal(str, object, bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.curves: dict[str, Pts] = {k: [[0.0, 0.0], [1.0, 1.0]] for k in CURVE_KEYS}
        self.hist: dict | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.tabs = QTabBar()
        self.tabs.setObjectName("chanTabs")
        self.tabs.setDrawBase(False)
        self.tabs.setExpanding(False)
        for _k, lbl, _c in CHANNELS:
            self.tabs.addTab(lbl)
        top.addWidget(self.tabs, 1)
        self.btn_reset = QToolButton()
        self.btn_reset.setText("↺")
        self.btn_reset.setToolTip("Restablecer la curva del canal")
        self.btn_reset.clicked.connect(self.reset_channel)
        top.addWidget(self.btn_reset)
        lay.addLayout(top)
        self.canvas = CurveCanvas()
        lay.addWidget(self.canvas)
        io = QHBoxLayout()
        io.setSpacing(6)
        self.in_spin = QSpinBox()
        self.out_spin = QSpinBox()
        for s in (self.in_spin, self.out_spin):
            s.setRange(0, 255)
            s.setEnabled(False)
            s.setKeyboardTracking(False)
        li, lo = QLabel("Entrada"), QLabel("Salida")
        li.setObjectName("muted")
        lo.setObjectName("muted")
        self.count_label = QLabel("")
        self.count_label.setObjectName("muted")
        io.addWidget(li)
        io.addWidget(self.in_spin)
        io.addWidget(lo)
        io.addWidget(self.out_spin)
        io.addStretch(1)
        io.addWidget(self.count_label)
        lay.addLayout(io)
        hint = QLabel("Click agrega · arrastrar mueve · doble click o arrastrar afuera borra")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.canvas.edited.connect(self._edited)
        self.canvas.selectionChanged.connect(self._sel_changed)
        self.in_spin.valueChanged.connect(self._io_changed)
        self.out_spin.valueChanged.connect(self._io_changed)
        self._tab_changed(0)

    @property
    def key(self) -> str:
        return CHANNELS[self.tabs.currentIndex()][0]

    def set_channel(self, key: str) -> None:
        self.tabs.setCurrentIndex([c[0] for c in CHANNELS].index(key))

    def set_curves(self, curves: dict[str, Any]) -> None:  # noqa: F821
        for k in CURVE_KEYS:
            if k in curves:
                self.curves[k] = _clean(curves[k])
        self._refresh(keep=True)

    def set_histogram(self, hist: dict | None) -> None:
        self.hist = hist
        self._refresh(keep=True)

    def _refresh(self, keep: bool) -> None:
        k = self.key
        col = CHANNELS[self.tabs.currentIndex()][2]
        self.canvas.color = col
        self.canvas.others = [(self.curves[kk], c) for kk, _l, c in CHANNELS
                              if kk != k and not (len(self.curves[kk]) == 2 and self.curves[kk] == [[0.0, 0.0], [1.0, 1.0]])]
        h = self.hist.get(HIST_KEY[k]) if isinstance(self.hist, dict) else None
        self.canvas.hist = None if h is None else np.asarray(h)
        self.canvas.set_points(self.curves[k], keep_selection=keep)
        self._sel_changed(self.canvas.selected)

    def _tab_changed(self, _i: int) -> None:
        self.canvas.selected = -1
        self._refresh(keep=False)

    def _edited(self, pts, final: bool) -> None:
        self.curves[self.key] = [list(p) for p in pts]
        self.count_label.setText(f"{len(pts)}/{CURVE_MAX_POINTS} puntos")
        self.changed.emit(self.key, [list(p) for p in pts], final)

    def _sel_changed(self, i: int) -> None:
        pts = self.canvas.points
        ok = 0 <= i < len(pts)
        for s in (self.in_spin, self.out_spin):
            s.blockSignals(True)
            s.setEnabled(ok)
        if ok:
            self.in_spin.setValue(round(pts[i][0] * 255))
            self.out_spin.setValue(round(pts[i][1] * 255))
        for s in (self.in_spin, self.out_spin):
            s.blockSignals(False)
        self.count_label.setText(f"{len(pts)}/{CURVE_MAX_POINTS} puntos")

    def _io_changed(self, _v: int) -> None:
        i = self.canvas.selected
        if i < 0:
            return
        pts = move_point(self.canvas.points, i, self.in_spin.value() / 255.0, self.out_spin.value() / 255.0)
        self.canvas.set_points(pts)
        self._edited(pts, True)
        self._sel_changed(i)

    def select_point(self, i: int) -> None:
        self.canvas.selected = i
        self._sel_changed(i)
        self.canvas.update()

    def reset_channel(self) -> None:
        pts = [[0.0, 0.0], [1.0, 1.0]]
        self.canvas.set_points(pts, keep_selection=False)
        self._edited(pts, True)
        self._sel_changed(-1)
