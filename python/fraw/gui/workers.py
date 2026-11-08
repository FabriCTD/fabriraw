"""Background threads: latest-request-wins renderer, one-shot tasks, thumbnail loader."""

from __future__ import annotations

import io
import os
import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QTransform

from .thumbs import peek_file

__all__ = ["LatestWorker", "TaskThread", "ThumbLoader", "orientation_transform"]


class LatestWorker(QThread):
    """Processes jobs one at a time; a new ``submit`` replaces any job still pending.

    So while the user drags a slider the worker always renders the *latest*
    state and never queues stale frames.
    """

    result = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[[Any], Any], name: str = "worker", parent=None):
        super().__init__(parent)
        self.setObjectName(name)
        self._fn = fn
        self._cond = threading.Condition()
        self._job: Any = None
        self._has_job = False
        self._stop = False
        self._busy = False

    def submit(self, job: Any) -> None:
        with self._cond:
            self._job = job
            self._has_job = True
            self._cond.notify()
        if not self.isRunning():
            self.start()

    def idle(self) -> bool:
        with self._cond:
            return not self._has_job and not self._busy

    def stop(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self.wait(5000)

    def run(self) -> None:  # noqa: D401 (Qt API)
        while True:
            with self._cond:
                while not self._has_job and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job, self._job, self._has_job = self._job, None, False
                self._busy = True
            try:
                r = self._fn(job)
            except Exception:  # noqa: BLE001 - reported to the UI
                r = None
                self.failed.emit(traceback.format_exc(limit=4))
            finally:
                with self._cond:
                    self._busy = False
            if r is not None:
                self.result.emit(r)


class TaskThread(QThread):
    """Runs ``fn(report, cancelled)`` once in background.

    ``report(i, text)`` emits ``progress``; ``cancelled()`` is True after ``cancel()``.
    """

    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, fn: Callable[[Callable[[int, str], None], Callable[[], bool]], Any], parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            r = self._fn(lambda i, t="": self.progress.emit(int(i), str(t)), lambda: self._cancel)
        except Exception as e:  # noqa: BLE001 - reported to the UI
            self.failed.emit(_describe(e))
            return
        self.done.emit(r)


def _describe(e: BaseException) -> str:
    from ..format import FrawError

    if isinstance(e, FrawError):
        return f"Archivo FabriRAW inválido ({e.code.value}): {e.msg}"
    if isinstance(e, FileNotFoundError):
        return f"No existe el archivo: {e.filename or e}"
    if isinstance(e, OSError):
        return f"Error de E/S: {e}"
    return f"{type(e).__name__}: {e}"


def orientation_transform(o: int) -> QTransform | None:
    """QTransform that displays an unrotated image upright (EXIF 1–8)."""
    o = int(o)
    return {
        2: QTransform(-1, 0, 0, 1, 0, 0),
        3: QTransform(-1, 0, 0, -1, 0, 0),
        4: QTransform(1, 0, 0, -1, 0, 0),
        5: QTransform(0, 1, 1, 0, 0, 0),
        6: QTransform().rotate(90),
        7: QTransform(0, -1, -1, 0, 0, 0),
        8: QTransform().rotate(270),
    }.get(o)


def _decode_thumb(fmt: int | None, size: tuple[int, int], data: bytes) -> QImage | None:
    if fmt == 0:
        img = QImage.fromData(data)
        return None if img.isNull() else img
    if fmt == 1 and len(data) == size[0] * size[1] * 3:
        return QImage(data, size[0], size[1], size[0] * 3, QImage.Format.Format_RGB888).copy()
    return None


FRAW_MAGIC = b"FRAW\r\n\x1a\n"
FSET_MAGIC = b"FSET\r\n\x1a\n"


def load_thumbnail(path: str, max_side: int = 160) -> tuple[QImage | None, dict]:
    """(thumbnail QImage, info) for a .fraw/.frw/.fset (THMB) or a camera RAW (embedded preview)."""
    with open(path, "rb") as fh:
        magic = fh.read(8)
    info: dict = {}
    img: QImage | None = None
    orientation = 1
    if magic in (FRAW_MAGIC, FSET_MAGIC):  # por contenido, no por extensión
        pk = peek_file(path)
        img = _decode_thumb(pk.thumb_format, pk.thumb_size, pk.thumb_data)
        orientation = pk.orientation
        info = {"meta": pk.meta or {}, "size": (pk.width, pk.height), "frames": pk.frames}
    else:
        import rawpy  # noqa: PLC0415

        with rawpy.imread(path) as raw:
            try:
                t = raw.extract_thumb()
            except rawpy.LibRawError:
                t = None
            orientation = {0: 1, 3: 3, 5: 8, 6: 6}.get(int(raw.sizes.flip), 1)
        if t is not None and "JPEG" in getattr(t.format, "name", str(t.format)).upper():
            img = QImage.fromData(bytes(t.data))
            orientation = 1  # embedded JPEG previews are usually already upright
        elif t is not None:
            from PIL import Image  # noqa: PLC0415

            buf = io.BytesIO()
            Image.fromarray(t.data).save(buf, "PNG")
            img = QImage.fromData(buf.getvalue())
        info = {"meta": {}, "raw": True}
    if img is not None and not img.isNull():
        tr = orientation_transform(orientation)
        if tr is not None:
            img = img.transformed(tr)
        from PySide6.QtCore import Qt  # noqa: PLC0415

        img = img.scaled(max_side, max_side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    return img, info


class ThumbLoader(QThread):
    """Loads library thumbnails in background; emits ``loaded(path, QImage|None, info)``."""

    loaded = Signal(str, object, object)

    def __init__(self, paths: list[str], max_side: int = 160, parent=None):
        super().__init__(parent)
        self.paths = list(paths)
        self.max_side = max_side
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        for p in self.paths:
            if self._cancel:
                return
            try:
                img, info = load_thumbnail(p, self.max_side)
            except Exception as e:  # noqa: BLE001 - a broken file just gets no thumbnail
                img, info = None, {"error": str(e)}
            self.loaded.emit(p, img, info)
