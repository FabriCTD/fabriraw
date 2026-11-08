"""Lienzo de la foto: QImage sin copia sobre la vista previa numpy, zoom/pan,
mosaico de detalle, Antes/Después (dividido con cortina o lado a lado), aviso
de recorte de luces/sombras, cuentagotas, ajuste dirigido y herramienta de recorte.

Coordenadas
-----------
* *espacio imagen*: píxeles de la imagen revelada a resolución completa tal como
  se muestra (``image_size`` = (w, h); con la herramienta de recorte activa es la
  imagen sin recortar).
* ``zoom``: píxeles físicos por píxel de imagen — ``1.0`` es "100 %".
* La vista previa se dibuja estirada sobre todo el rectángulo de la imagen; el
  mosaico de detalle (render a resolución real del área visible) va encima.
* En modo "lado a lado" cada mitad del widget es una vista con el mismo zoom.

Las QImage envuelven directamente los buffers numpy (sin copia); el lienzo
mantiene referencias a los arrays mientras se usan.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from . import croptool

__all__ = ["ImageCanvas", "qimage_from_rgb8", "qimage_from_rgba8", "clip_overlay"]

ZOOM_MIN = 0.02
ZOOM_MAX = 16.0
COMPARE_MODES = ("off", "split", "side")


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


def qimage_from_rgba8(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, int(arr.strides[0]), QImage.Format.Format_RGBA8888)


def clip_overlay(shadows: np.ndarray, highlights: np.ndarray, show_sh: bool = True, show_hl: bool = True) -> np.ndarray:
    """Máscaras booleanas → RGBA (luces rojo, sombras azul)."""
    h, w = highlights.shape
    out = np.zeros((h, w, 4), np.uint8)
    if show_sh:
        out[shadows] = (40, 110, 255, 230)
    if show_hl:
        out[highlights] = (255, 40, 40, 230)
    return out


class ImageCanvas(QWidget):
    zoomChanged = Signal(float)
    viewChanged = Signal()
    picked = Signal(float, float)  # espacio imagen (cuentagotas)
    hovered = Signal(float, float)  # espacio imagen; (-1, -1) al salir
    splitChanged = Signal(float)
    cropChanged = Signal(object)  # (l, t, r, b) normalizado sobre la imagen sin recortar
    straighten = Signal(float, bool)  # delta de ángulo (°) desde el inicio del arrastre, final
    cropAccepted = Signal()
    cropCancelled = Signal()
    targetDrag = Signal(float, float, float, bool)  # x, y (imagen), desplazamiento vertical px (+ arriba), final

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(320, 240)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._img: QImage | None = None
        self._buf: np.ndarray | None = None  # mantiene viva la memoria de la QImage
        self._tile: QImage | None = None
        self._tile_buf: np.ndarray | None = None
        self._tile_rect = QRectF()
        self._before: QImage | None = None
        self._before_buf: np.ndarray | None = None
        self._overlay: QImage | None = None
        self._overlay_buf: np.ndarray | None = None
        self.image_size: tuple[int, int] = (0, 0)
        self.zoom = 1.0
        self.offset = QPointF(0, 0)  # px lógicos del origen de la imagen (en la vista)
        self.fit_mode = True
        self.tool = "none"  # none | picker | crop | target
        self.compare = "off"  # off | split | side
        self.split = 0.5
        self.show_before_label = ""
        self._drag: QPointF | None = None
        self._split_drag = False
        self.placeholder = "Abrí un archivo .fraw / .frw / .fset o importá un RAW de cámara"
        self.paint_count = 0
        # recorte
        self.crop_rect: croptool.Rect = (0.0, 0.0, 1.0, 1.0)
        self.crop_aspect: float | None = None
        self.crop_angle = 0.0
        self._crop_op: str | None = None  # handle | "move" | "rotate"
        self._crop_start: QPointF | None = None
        self._crop_rect0: croptool.Rect = self.crop_rect
        self._rot_a0 = 0.0
        # ajuste dirigido
        self._target_start: QPointF | None = None
        self._target_img: QPointF | None = None

    # ---------------------------------------------------------------- state
    @property
    def picker(self) -> bool:  # compatibilidad
        return self.tool == "picker"

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
            self.set_overlay(None)
            self.image_size = (0, 0)
            self.update()
            return
        new_size = image_size or (arr.shape[1], arr.shape[0])
        first = self._img is None or tuple(new_size) != tuple(self.image_size)
        self._buf = arr
        self._img = qimage_from_rgb8(arr)
        self.image_size = (int(new_size[0]), int(new_size[1]))
        if first:
            self.clear_tile()
            self.fit_mode = True
            self.fit()
        self.update()

    def set_before(self, arr: np.ndarray | None) -> None:
        self._before_buf = arr
        self._before = None if arr is None else qimage_from_rgb8(arr)
        self.update()

    def set_overlay(self, rgba: np.ndarray | None) -> None:
        self._overlay_buf = rgba
        self._overlay = None if rgba is None else qimage_from_rgba8(rgba)
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

    def set_tool(self, tool: str) -> None:
        self.tool = tool
        cur = {"picker": Qt.CursorShape.CrossCursor, "target": Qt.CursorShape.SizeVerCursor,
               "crop": Qt.CursorShape.ArrowCursor}.get(tool, Qt.CursorShape.ArrowCursor)
        self.setCursor(QCursor(cur))
        if tool == "crop":
            self.setFocus()
        self.update()

    def set_picker(self, on: bool) -> None:  # compatibilidad
        self.set_tool("picker" if on else ("none" if self.tool == "picker" else self.tool))

    def set_compare(self, mode: str) -> None:
        self.compare = mode if mode in COMPARE_MODES else "off"
        if self.fit_mode:
            self._set_zoom(self.fit_zoom(), center=True)
        else:
            self._clamp_offset()
            self.viewChanged.emit()
        self.update()

    def set_crop(self, rect: croptool.Rect, aspect: float | None, angle: float) -> None:
        self.crop_rect = tuple(float(v) for v in rect)  # type: ignore[assignment]
        self.crop_aspect = aspect
        self.crop_angle = float(angle)
        self.update()

    # ------------------------------------------------------------ geometry
    def view_width(self) -> float:
        return self.width() / 2.0 if self.compare == "side" else float(self.width())

    def _scale(self) -> float:
        """Logical px per image px."""
        return self.zoom / self.dpr

    def image_rect(self) -> QRectF:
        s = self._scale()
        return QRectF(self.offset.x(), self.offset.y(), self.image_size[0] * s, self.image_size[1] * s)

    def _local(self, p: QPointF) -> QPointF:
        if self.compare == "side" and p.x() >= self.view_width():
            return QPointF(p.x() - self.view_width(), p.y())
        return p

    def to_image(self, p: QPointF) -> QPointF:
        p = self._local(p)
        s = self._scale()
        return QPointF((p.x() - self.offset.x()) / s, (p.y() - self.offset.y()) / s)

    def to_view(self, x: float, y: float) -> QPointF:
        s = self._scale()
        return QPointF(self.offset.x() + x * s, self.offset.y() + y * s)

    def visible_image_rect(self) -> QRectF:
        """Visible part of the image, in image-space pixels (may be empty)."""
        if not self.image_size[0]:
            return QRectF()
        s = self._scale()
        a = QPointF(-self.offset.x() / s, -self.offset.y() / s)
        b = QPointF((self.view_width() - self.offset.x()) / s, (self.height() - self.offset.y()) / s)
        return QRectF(a, b).intersected(QRectF(0, 0, *self.image_size))

    def fit_zoom(self) -> float:
        w, h = self.image_size
        if not w or not h:
            return 1.0
        m = 16 if self.tool != "crop" else 28
        return max(ZOOM_MIN, min((self.view_width() - 2 * m) / w, (self.height() - 2 * m) / h) * self.dpr)

    def fit(self) -> None:
        self.fit_mode = True
        self._set_zoom(self.fit_zoom(), center=True)

    def zoom_100(self) -> None:
        self.zoom_to(1.0)

    def zoom_to(self, z: float) -> None:
        self.fit_mode = False
        self.zoom_at(z, QPointF(self.view_width() / 2, self.height() / 2), absolute=True)

    def zoom_at(self, factor: float, pos: QPointF, absolute: bool = False) -> None:
        if not self.image_size[0]:
            return
        self.fit_mode = False
        pos = self._local(pos)
        s0 = self._scale()
        anchor = QPointF((pos.x() - self.offset.x()) / s0, (pos.y() - self.offset.y()) / s0)
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
            self.offset = QPointF((self.view_width() - r.width()) / 2, (self.height() - r.height()) / 2)
        self._clamp_offset()
        self.update()
        self.zoomChanged.emit(self.zoom)
        self.viewChanged.emit()

    def _clamp_offset(self) -> None:
        r = self.image_rect()
        vw = self.view_width()
        x, y = self.offset.x(), self.offset.y()
        if r.width() <= vw:
            x = (vw - r.width()) / 2
        else:
            x = min(0.0, max(vw - r.width(), x))
        if r.height() <= self.height():
            y = (self.height() - r.height()) / 2
        else:
            y = min(0.0, max(self.height() - r.height(), y))
        self.offset = QPointF(x, y)

    # ------------------------------------------------------- crop geometry
    def _crop_view_rect(self) -> QRectF:
        l, t, r, b = self.crop_rect
        w, h = self.image_size
        a, c = self.to_view(l * w, t * h), self.to_view(r * w, b * h)
        return QRectF(a, c)

    def _crop_hit(self, p: QPointF) -> str | None:
        r = self._crop_view_rect()
        pts = {"tl": r.topLeft(), "t": QPointF(r.center().x(), r.top()), "tr": r.topRight(),
               "r": QPointF(r.right(), r.center().y()), "br": r.bottomRight(),
               "b": QPointF(r.center().x(), r.bottom()), "bl": r.bottomLeft(), "l": QPointF(r.left(), r.center().y())}
        for k, q in pts.items():
            if abs(q.x() - p.x()) <= 9 and abs(q.y() - p.y()) <= 9:
                return k
        if r.contains(p):
            return "move"
        return "rotate"

    def _norm(self, p: QPointF) -> tuple[float, float]:
        q = self.to_image(p)
        w, h = self.image_size
        return q.x() / max(1, w), q.y() / max(1, h)

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

    def _split_x(self) -> float:
        r = self.image_rect()
        return r.left() + r.width() * self.split

    def mousePressEvent(self, e):  # noqa: N802
        if not self.has_image():
            return
        pos = e.position()
        left = e.button() == Qt.MouseButton.LeftButton
        if left and self.tool == "picker":
            p = self.to_image(pos)
            if 0 <= p.x() < self.image_size[0] and 0 <= p.y() < self.image_size[1]:
                self.picked.emit(p.x(), p.y())
            return
        if left and self.tool == "crop":
            self._crop_op = self._crop_hit(pos)
            self._crop_start = pos
            self._crop_rect0 = self.crop_rect
            c = self._crop_view_rect().center()
            self._rot_a0 = math.degrees(math.atan2(pos.y() - c.y(), pos.x() - c.x()))
            return
        if left and self.tool == "target":
            p = self.to_image(pos)
            self._target_start = pos
            self._target_img = p
            return
        if left and self.compare == "split" and abs(pos.x() - self._split_x()) < 8:
            self._split_drag = True
            return
        if e.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
            self._drag = pos
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

    def mouseMoveEvent(self, e):  # noqa: N802
        pos = e.position()
        if self.has_image():
            p = self.to_image(pos)
            if 0 <= p.x() < self.image_size[0] and 0 <= p.y() < self.image_size[1]:
                self.hovered.emit(p.x(), p.y())
            else:
                self.hovered.emit(-1.0, -1.0)
        if self.tool == "crop" and self._crop_op is not None:
            self._crop_drag(pos, False)
            return
        if self.tool == "crop":
            op = self._crop_hit(pos)
            cur = {"tl": Qt.CursorShape.SizeFDiagCursor, "br": Qt.CursorShape.SizeFDiagCursor,
                   "tr": Qt.CursorShape.SizeBDiagCursor, "bl": Qt.CursorShape.SizeBDiagCursor,
                   "t": Qt.CursorShape.SizeVerCursor, "b": Qt.CursorShape.SizeVerCursor,
                   "l": Qt.CursorShape.SizeHorCursor, "r": Qt.CursorShape.SizeHorCursor,
                   "move": Qt.CursorShape.SizeAllCursor}.get(op or "", Qt.CursorShape.CrossCursor)
            self.setCursor(QCursor(cur))
            return
        if self.tool == "target" and self._target_start is not None and self._target_img is not None:
            self.targetDrag.emit(self._target_img.x(), self._target_img.y(), self._target_start.y() - pos.y(), False)
            return
        if self._split_drag:
            r = self.image_rect()
            self.split = min(1.0, max(0.0, (pos.x() - r.left()) / max(1.0, r.width())))
            self.splitChanged.emit(self.split)
            self.update()
            return
        if self.compare == "split" and self.tool == "none" and self._drag is None:
            near = abs(pos.x() - self._split_x()) < 8
            self.setCursor(QCursor(Qt.CursorShape.SplitHCursor if near else Qt.CursorShape.ArrowCursor))
        if self._drag is not None:
            d = pos - self._drag
            self._drag = pos
            self.offset += d
            self._clamp_offset()
            self.update()
            self.viewChanged.emit()

    def _crop_drag(self, pos: QPointF, final: bool) -> None:
        op = self._crop_op
        w, h = self.image_size
        if op == "rotate":
            c = self._crop_view_rect().center()
            a = math.degrees(math.atan2(pos.y() - c.y(), pos.x() - c.x()))
            d = (a - self._rot_a0 + 180) % 360 - 180
            self.straighten.emit(d, final)
            return
        if op == "move" and self._crop_start is not None:
            s = self._scale()
            dx = (pos.x() - self._crop_start.x()) / s / max(1, w)
            dy = (pos.y() - self._crop_start.y()) / s / max(1, h)
            self.crop_rect = croptool.move_rect(self._crop_rect0, dx, dy)
        elif op in croptool.HANDLES:
            x, y = self._norm(pos)
            self.crop_rect = croptool.drag_handle(self.crop_rect, op, x, y, self.crop_aspect, w, h)
        self.cropChanged.emit(self.crop_rect)
        self.update()

    def mouseReleaseEvent(self, e):  # noqa: N802
        if self.tool == "crop" and self._crop_op is not None:
            self._crop_drag(e.position(), True)
            self._crop_op = None
            self.update()
            return
        if self.tool == "target" and self._target_start is not None and self._target_img is not None:
            self.targetDrag.emit(self._target_img.x(), self._target_img.y(), self._target_start.y() - e.position().y(), True)
            self._target_start = self._target_img = None
            return
        if self._split_drag:
            self._split_drag = False
            return
        if self._drag is not None:
            self._drag = None
            self.set_tool(self.tool)

    def mouseDoubleClickEvent(self, e):  # noqa: N802
        if not self.has_image() or self.tool in ("picker", "target"):
            return
        if self.tool == "crop":
            self.cropAccepted.emit()
            return
        if self.fit_mode:
            self.zoom_at(1.0, e.position(), absolute=True)
        else:
            self.fit()

    def leaveEvent(self, _e):  # noqa: N802
        self.hovered.emit(-1.0, -1.0)

    def keyPressEvent(self, e):  # noqa: N802
        if self.tool == "crop" and e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.cropAccepted.emit()
            return
        if self.tool == "crop" and e.key() == Qt.Key.Key_Escape:
            self.cropCancelled.emit()
            return
        super().keyPressEvent(e)

    # --------------------------------------------------------------- paint
    def _draw(self, p: QPainter, img: QImage, rect: QRectF, clip: QRectF) -> None:
        """Draws ``img`` stretched over image-space ``rect``, only the visible part."""
        s = self._scale()
        target = QRectF(self.offset.x() + rect.x() * s, self.offset.y() + rect.y() * s, rect.width() * s, rect.height() * s)
        vis = target.intersected(clip)
        if vis.isEmpty():
            return
        sx = img.width() / target.width()
        sy = img.height() / target.height()
        src = QRectF((vis.x() - target.x()) * sx, (vis.y() - target.y()) * sy, vis.width() * sx, vis.height() * sy)
        p.drawImage(vis, img, src)

    def _draw_after(self, p: QPainter, clip: QRectF) -> None:
        full = QRectF(0, 0, *self.image_size)
        smooth = self.zoom * self.image_size[0] / max(1, self._img.width()) < 2.5  # type: ignore[union-attr]
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, smooth)
        self._draw(p, self._img, full, clip)  # type: ignore[arg-type]
        if self._tile is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom < 1.0)
            self._draw(p, self._tile, self._tile_rect, clip)
        if self._overlay is not None:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            self._draw(p, self._overlay, full, clip)

    def _draw_before(self, p: QPainter, clip: QRectF) -> None:
        img = self._before or self._img
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._draw(p, img, QRectF(0, 0, *self.image_size), clip)  # type: ignore[arg-type]

    def _badge(self, p: QPainter, text: str, x: float, y: float, right: bool = False) -> None:
        f = QFont(p.font())
        f.setPointSizeF(8.5)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        w = fm.horizontalAdvance(text) + 14
        r = QRectF(x - w if right else x, y, w, 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(9, 8, 12, 190))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QColor("#EDEBF0"))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

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
        vw = self.view_width()
        if self.compare == "side":
            left = QRectF(0, 0, vw, self.height())
            p.setClipRect(left)
            p.setPen(QPen(QColor("#2E2D33"), 1))
            p.drawRect(r.adjusted(-1, -1, 0, 0))
            self._draw_before(p, left)
            self._badge(p, "ANTES", 10, 10)
            p.setClipping(False)
            p.translate(vw, 0)
            right = QRectF(0, 0, vw, self.height())
            p.setClipRect(right)
            p.setPen(QPen(QColor("#2E2D33"), 1))
            p.drawRect(r.adjusted(-1, -1, 0, 0))
            self._draw_after(p, right)
            self._badge(p, "DESPUÉS", vw - 10, 10, right=True)
            p.setClipping(False)
            p.translate(-vw, 0)
            p.setPen(QPen(QColor("#45434C"), 1))
            p.drawLine(QPointF(vw, 0), QPointF(vw, self.height()))
        else:
            p.setPen(QPen(QColor("#2E2D33"), 1))
            p.drawRect(r.adjusted(-1, -1, 0, 0))
            whole = QRectF(self.rect())
            self._draw_after(p, whole)
            if self.compare == "split":
                xs = self._split_x()
                lclip = QRectF(0, 0, max(0.0, xs), self.height())
                self._draw_before(p, lclip.intersected(whole))
                p.setPen(QPen(QColor("#EDEBF0"), 1.5))
                p.drawLine(QPointF(xs, r.top()), QPointF(xs, r.bottom()))
                cy = max(r.top(), 0) + min(r.height(), self.height()) / 2
                p.setBrush(QColor("#F0B232"))
                p.setPen(QPen(QColor("#09080C"), 1.5))
                p.drawEllipse(QPointF(xs, cy), 9, 9)
                p.setPen(QPen(QColor("#09080C"), 1.6))
                p.drawLine(QPointF(xs - 4, cy), QPointF(xs + 4, cy))
                top = max(r.top(), 0) + 10
                self._badge(p, "ANTES", xs - 10, top, right=True)
                self._badge(p, "DESPUÉS", xs + 10, top)
            if self.show_before_label:
                self._badge(p, self.show_before_label, 12, 12)
        if self.tool == "crop":
            self._paint_crop(p)
        p.end()

    def _paint_crop(self, p: QPainter) -> None:
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        cr = self._crop_view_rect()
        shade = QPainterPath()
        shade.addRect(QRectF(self.rect()))
        inner = QPainterPath()
        inner.addRect(cr)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(9, 8, 12, 165))
        p.drawPath(shade.subtracted(inner))
        p.setPen(QPen(QColor(237, 235, 240, 120), 1))
        for i in (1, 2):
            x = cr.left() + cr.width() * i / 3
            y = cr.top() + cr.height() * i / 3
            p.drawLine(QPointF(x, cr.top()), QPointF(x, cr.bottom()))
            p.drawLine(QPointF(cr.left(), y), QPointF(cr.right(), y))
        p.setPen(QPen(QColor("#EDEBF0"), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(cr)
        p.setBrush(QColor("#F0B232"))
        p.setPen(QPen(QColor("#09080C"), 1))
        for q in (cr.topLeft(), cr.topRight(), cr.bottomLeft(), cr.bottomRight(),
                  QPointF(cr.center().x(), cr.top()), QPointF(cr.center().x(), cr.bottom()),
                  QPointF(cr.left(), cr.center().y()), QPointF(cr.right(), cr.center().y())):
            p.drawRect(QRectF(q.x() - 4, q.y() - 4, 8, 8))
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.image_size
        l, t, rr, b = self.crop_rect
        txt = f"{(rr - l) * w:.0f} × {(b - t) * h:.0f} px · {self.crop_angle:+.2f}°"
        self._badge(p, txt, cr.center().x() - 70, min(self.height() - 26, cr.bottom() + 8))
        self._badge(p, "RECORTE · Enter aplica · Esc cancela · arrastrá afuera para enderezar", 12, 12)
