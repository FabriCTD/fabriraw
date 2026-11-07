"""Rasterizes packaging/icon.svg (same geometry, drawn with Pillow) into
python/fraw/gui/icon.png (256 px) and packaging/icon.ico (16-256 px).

Pillow cannot read SVG, so the shapes of icon.svg are reproduced here 1:1
(64x64 viewBox) and supersampled 8x for clean anti-aliased edges.

    python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SS = 8  # supersampling


def render(size: int) -> Image.Image:
    big = size * SS
    k = big / 64.0
    im = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    def box(x, y, w, h):
        return [x * k, y * k, (x + w) * k - 1, (y + h) * k - 1]

    # rounded square with 2px stroke: outer stroke colour, inner fill
    d.rounded_rectangle(box(0, 0, 64, 64), radius=15 * k, fill="#2e2d33")
    d.rounded_rectangle(box(2, 2, 60, 60), radius=13 * k, fill="#09080c")
    # "F" (M13 11h29v9H23v7h16v8H23v18H13z)
    pts = [(13, 11), (42, 11), (42, 20), (23, 20), (23, 27), (39, 27), (39, 35), (23, 35), (23, 53), (13, 53)]
    d.polygon([(x * k, y * k) for x, y in pts], fill="#f0b232")
    # Bayer RGGB
    for (x, y, c) in ((36, 38, "#e5484d"), (45, 38, "#23a559"), (36, 47, "#23a559"), (45, 47, "#4c8dff")):
        d.rounded_rectangle(box(x, y, 8, 8), radius=1.5 * k, fill=c)
    return im.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    png = ROOT / "python" / "fraw" / "gui" / "icon.png"
    ico = ROOT / "packaging" / "icon.ico"
    render(256).save(png)
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    imgs = [render(s) for s in sizes]
    imgs[-1].save(ico, format="ICO", sizes=[(s, s) for s in sizes], append_images=imgs[:-1])
    print(f"{png} ({png.stat().st_size} B)\n{ico} ({ico.stat().st_size} B)")


if __name__ == "__main__":
    main()
