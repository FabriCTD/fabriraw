"""Preview canvas: zero-copy QImage over the numpy preview, zoom/pan, detail tile.

Coordinates
-----------
* *image space*: pixels of the full-resolution, oriented developed image
  (what ``develop()`` would return). ``image_size`` is its (w, h).
* ``zoom``: device (physical) pixels per image pixel — ``1.0`` is "100 %".
* The base preview (``processor.preview``) is drawn stretched over the whole
  image rectangle; an optional *detail tile* (full-resolution develop of the
  visible area, rendered in background when zoomed in) is drawn on top at its
  image-space rectangle.

The QImage objects wrap the numpy buffers directly (no copy); the arrays are
kept referenced by the canvas for as long as the QImage is in use.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

__all__ = ["ImageCanvas", "qimage_from_rgb8"]

ZOOM_MIN = 0.02
ZOOM_MAX = 16.0


def qimage_from_rgb8(arr: np.ndarray) -> QImage:
    """QImage view (Format_RGB888) over a C-contiguous H×W×3 uint8 array — no copy.

    The caller MUST keep ``arr`` alive while the QImage is used.
    """
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("expected H×W×3 uint8")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_RGB888)


class ImageCanvas(QWidget):
    zoomChanged = Signal(float)
    viewChanged = Signal()
    picked = Signal(float, float)  # image-space coordinates (eyedropper)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._img: QImage | None = None
        self._buf: np.ndarray | None = None  # keeps the QImage memory alive
        self._tile: QImage | None = None
        self._tile_buf: np.ndarray | None = None
        self._tile_rect = QRectF()
        self.image_size: tuple[int, int] = (0, 0)
        self.zoom = 1.0
        self.offset = QPointF(0, 0)  # logical px of the image origin
        self.fit_mode = True
        self.picker = False
        self._drag: QPointF | None = None
        self.placeholder = "Abrí un archivo .fraw / .frw / .fset o importá un RAW de cámara"
        self.paint_count = 0

    # ---------------------------------------------------------------- state
    @property
    def dpr(self) -> float:
        return float(self.devicePixelRatioF() or 1.0)

    def has_image(self) -> bool:
        return self._img is not None

    def current_qimage(self) -> QImage | None:
        return self._img

    def set_image(self, arr: np.ndarray | None, image_size: tuple[int, int] | None = None) -> None:
        if arr is None:
            self._img = self._buf = None
            self.clear_tile()
            self.image_size = (0, 0)
            self.update()
            return
        new_size = image_size or (arr.shape[1], arr.shape[0])
        first = self._img is None or new_size != self.image_size
        self._buf = arr
        self._img = qimage_from_rgb8(arr)
        self.image_size = (int(new_size[0]), int(new_size[1]))
        if first:
            self.clear_tile()
            self.fit_mode = True
            self.fit()
        self.update()

    def set_tile(self, arr: np.ndarray, rect: QRectF) -> None:
        self._tile_buf = arr
        self._tile = qimage_from_rgb8(arr)
        self._tile_rect = QRectF(rect)
        self.update()

    def clear_tile(self) -> None:
        if self._tile is not None:
            self._tile = self._tile_buf = None
            self._tile_rect = QRectF()
            self.update()

    def has_tile(self) -> bool:
        return self._tile is not None

    def set_picker(self, on: bool) -> None:
        self.picker = on
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor))

    # ------------------------------------------------------------ geometry
    def _scale(self) -> float:
        """Logical px per image px."""
        return self.zoom / self.dpr

    def image_rect(self) -> QRectF:
        s = self._scale()
        return QRectF(self.offset.x(), self.offset.y(), self.image_size[0] * s, self.image_size[1] * s)

    def to_image(self, p: QPointF) -> QPointF:
        s = self._scale()
        return QPointF((p.x() - self.offset.x()) / s, (p.y() - self.offset.y()) / s)

    def visible_image_rect(self) -> QRectF:
        """Visible part of the image, in image-space pixels (may be empty)."""
        if not self.image_size[0]:
            return QRectF()
        a = self.to_image(QPointF(0, 0))
        b = self.to_image(QPointF(self.width(), self.height()))
        return QRectF(a, b).intersected(QRectF(0, 0, *self.image_size))

    def fit_zoom(self) -> float:
        w, h = self.image_size
        if not w or not h:
            return 1.0
        m = 16
        return max(ZOOM_MIN, min((self.width() - 2 * m) / w, (self.height() - 2 * m) / h) * self.dpr)

    def fit(self) -> None:
        self.fit_mode = True
        self._set_zoom(self.fit_zoom(), center=True)

    def zoom_100(self) -> None:
        self.fit_mode = False
        self.zoom_at(1.0, QPointF(self.width() / 2, self.height() / 2), absolute=True)

    def zoom_at(self, factor: float, pos: QPointF, absolute: bool = False) -> None:
        if not self.image_size[0]:
            return
        self.fit_mode = False
        anchor = self.to_image(pos)
        z = factor if absolute else self.zoom * factor
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, z))
        s = self._scale()
        self.offset = QPointF(pos.x() - anchor.x() * s, pos.y() - anchor.y() * s)
        self._clamp_offset()
        self.update()
        self.zoomChanged.emit(self.zoom)
        self.viewChanged.emit()

    def _set_zoom(self, z: float, center: bool) -> None:
        self.zoom = min(ZOOM_MAX, max(ZOOM_MIN, z))
        if center:
            r = self.image_rect()
            self.offset = QPointF((self.width() - r.width()) / 2, (self.height() - r.height()) / 2)
        self._clamp_offset()
        self.update()
        self.zoomChanged.emit(self.zoom)
        self.viewChanged.emit()

    def _clamp_offset(self) -> None:
        r = self.image_rect()
        x, y = self.offset.x(), self.offset.y()
        if r.width() <= self.width():
            x = (self.width() - r.width()) / 2
        else:
            x = min(0.0, max(self.width() - r.width(), x))
        if r.height() <= self.height():
            y = (self.height() - r.height()) / 2
        else:
            y = min(0.0, max(self.height() - r.height(), y))
        self.offset = QPointF(x, y)

    # -------------------------------------------------------------- events
    def resizeEvent(self, e):  # noqa: N802
        if self.fit_mode:
            self._set_zoom(self.fit_zoom(), center=True)
        else:
            self._clamp_offset()
            self.viewChanged.emit()
        super().resizeEvent(e)

    def wheelEvent(self, e):  # noqa: N802
        if not self.has_image():
            return
        steps = e.angleDelta().y() / 120.0
        if steps:
            self.zoom_at(1.25**steps, e.position())
        e.accept()

    def mousePressEvent(self, e):  # noqa: N802
        if not self.has_image():
            return
        if e.button() == Qt.MouseButton.LeftButton and self.picker:
            p = self.to_image(e.position())
            if 0 <= p.x() < self.image_size[0] and 0 <= p.y() < self.image_size[1]:
                self.picked.emit(p.x(), p.y())
            return
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = e.position()
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, e):  # noqa: N802
        if self._drag is not None:
            d = e.position() - self._drag
            self._drag = e.position()
            self.offset += d
            self._clamp_offset()
            self.update()
            self.viewChanged.emit()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self._drag is not None:
            self._drag = None
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor if self.picker else Qt.CursorShape.ArrowCursor))

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if not self.has_image() or self.picker:
            return
        if self.fit_mode:
            self.zoom_at(1.0, e.position(), absolute=True)
        else:
            self.fit()

    # --------------------------------------------------------------- paint
    def _draw(self, p: QPainter, img: QImage, rect: QRectF) -> None:
        """Draws ``img`` stretched over image-space ``rect``, only the visible part."""
        s = self._scale()
        target = QRectF(self.offset.x() + rect.x() * s, self.offset.y() + rect.y() * s, rect.width() * s, rect.height() * s)
        vis = target.intersected(QRectF(self.rect()))
        if vis.isEmpty():
            return
        sx = img.width() / target.width()
        sy = img.height() / target.height()
        src = QRectF((vis.x() - target.x()) * sx, (vis.y() - target.y()) * sy, vis.width() * sx, vis.height() * sy)
        p.drawImage(vis, img, src)

    def paintEvent(self, _e):  # noqa: N802
        self.paint_count += 1
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#09080C"))
        if self._img is None:
            p.setPen(QColor("#A6A3AD"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
            p.end()
            return
        r = self.image_rect()
        p.setPen(QPen(QColor("#2E2D33"), 1))
        p.drawRect(r.adjusted(-1, -1, 0, 0))
        smooth = self.zoom * self.image_size[0] / max(1, self._img.width()) < 2.5
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        self._draw(p, self._img, QRectF(0, 0, *self.image_size))
        if self._tile is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1.0)
            self._draw(p, self._tile, self._tile_rect)
        p.end()
