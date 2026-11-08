"""Geometría del recorte (sin Qt). Rectángulos normalizados ``(l, t, r, b)`` en [0, 1]
sobre la imagen sin recortar de tamaño ``(W, H)`` píxeles. La proporción es
ancho:alto *en píxeles* (``aspect = w_px / h_px``)."""

from __future__ import annotations

Rect = tuple[float, float, float, float]

__all__ = ["aspect_value", "fit_aspect", "drag_handle", "move_rect", "rotate_rect_90", "flip_rect",
           "rect_aspect", "MIN_SIZE", "HANDLES"]

MIN_SIZE = 0.02
HANDLES = ("tl", "t", "tr", "r", "br", "b", "bl", "l")


def aspect_value(choice: str, w: int, h: int) -> float | None:
    if choice == "free":
        return None
    if choice == "original":
        return w / max(1, h)
    try:
        a, b = choice.split(":")
        return float(a) / float(b)
    except (ValueError, ZeroDivisionError):
        return None


def rect_aspect(rect: Rect, w: int, h: int) -> float:
    l, t, r, b = rect
    return (r - l) * w / max(1e-9, (b - t) * h)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def fit_aspect(rect: Rect, aspect: float | None, w: int, h: int) -> Rect:
    """Mayor rectángulo con ``aspect`` centrado en ``rect`` y contenido en él y en la imagen."""
    if aspect is None:
        return rect
    l, t, r, b = rect
    cx, cy = (l + r) / 2, (t + b) / 2
    wp, hp = (r - l) * w, (b - t) * h
    if wp / max(1e-9, hp) > aspect:
        wp = hp * aspect
    else:
        hp = wp / aspect
    nw, nh = wp / w, hp / h
    l2, t2 = cx - nw / 2, cy - nh / 2
    l2 = min(max(0.0, l2), 1.0 - nw)
    t2 = min(max(0.0, t2), 1.0 - nh)
    return (l2, t2, l2 + nw, t2 + nh)


def move_rect(rect: Rect, dx: float, dy: float) -> Rect:
    l, t, r, b = rect
    w, h = r - l, b - t
    l = min(max(0.0, l + dx), 1.0 - w)
    t = min(max(0.0, t + dy), 1.0 - h)
    return (l, t, l + w, t + h)


def drag_handle(rect: Rect, handle: str, x: float, y: float, aspect: float | None, w: int, h: int) -> Rect:
    """Mueve la manija ``handle`` hasta (x, y) normalizado respetando ``aspect``."""
    l, t, r, b = rect
    x, y = _clamp01(x), _clamp01(y)
    if aspect is None:
        if "l" in handle:
            l = min(x, r - MIN_SIZE)
        if "r" in handle:
            r = max(x, l + MIN_SIZE)
        if handle in ("t", "tl", "tr"):
            t = min(y, b - MIN_SIZE)
        if handle in ("b", "bl", "br"):
            b = max(y, t + MIN_SIZE)
        return (l, t, r, b)
    if len(handle) == 2:  # esquina: ancla en la opuesta
        ax = r if "l" in handle else l
        ay = b if handle[0] == "t" else t
        wp, hp = abs(x - ax) * w, abs(y - ay) * h
        if wp / max(1e-9, hp) > aspect:
            hp = wp / aspect
        else:
            wp = hp * aspect
        sx = -1 if "l" in handle else 1
        sy = -1 if handle[0] == "t" else 1
        max_w = (ax if sx < 0 else 1 - ax) * w
        max_h = (ay if sy < 0 else 1 - ay) * h
        k = min(1.0, max_w / max(1e-9, wp), max_h / max(1e-9, hp))
        wp, hp = max(wp * k, MIN_SIZE * w), max(hp * k, MIN_SIZE * h)
        if wp / hp > aspect:
            wp = hp * aspect
        else:
            hp = wp / aspect
        nx, ny = ax + sx * wp / w, ay + sy * hp / h
        return (min(ax, nx), min(ay, ny), max(ax, nx), max(ay, ny))
    # borde: cambia una dimensión, la otra sigue la proporción centrada
    cx, cy = (l + r) / 2, (t + b) / 2
    if handle in ("l", "r"):
        ax = r if handle == "l" else l
        wp = max(MIN_SIZE * w, abs(x - ax) * w)
        hp = wp / aspect
        max_h = 2 * min(cy, 1 - cy) * h
        max_w = (ax if handle == "l" else 1 - ax) * w
        k = min(1.0, max_h / hp, max_w / wp)
        wp, hp = wp * k, hp * k
        nx = ax - wp / w if handle == "l" else ax + wp / w
        return (min(ax, nx), cy - hp / h / 2, max(ax, nx), cy + hp / h / 2)
    ay = b if handle == "t" else t
    hp = max(MIN_SIZE * h, abs(y - ay) * h)
    wp = hp * aspect
    max_w = 2 * min(cx, 1 - cx) * w
    max_h = (ay if handle == "t" else 1 - ay) * h
    k = min(1.0, max_w / wp, max_h / hp)
    wp, hp = wp * k, hp * k
    ny = ay - hp / h if handle == "t" else ay + hp / h
    return (cx - wp / w / 2, min(ay, ny), cx + wp / w / 2, max(ay, ny))


def rotate_rect_90(rect: Rect, clockwise: bool) -> Rect:
    l, t, r, b = rect
    if clockwise:
        return (1 - b, l, 1 - t, r)
    return (t, 1 - r, b, 1 - l)


def flip_rect(rect: Rect, horizontal: bool) -> Rect:
    l, t, r, b = rect
    return (1 - r, t, 1 - l, b) if horizontal else (l, 1 - b, r, 1 - t)
