"""Plain-SVG plotting helpers shared by the figure scripts (no matplotlib).

The figures are written as SVG strings and rendered to PNG/PDF with rsvg-convert, falling back
to ImageMagick (render_svg).
"""
import subprocess
import shutil
import json

from src.utils.paths import repo_path

# ---- shared palette ----
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
GRID = "#e4e3de"

# categorical colours (blue, orange, aqua, yellow), chosen to stay distinguishable under colour-vision deficiency
CATEGORICAL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# diverging poles for cosine-similarity-style [-1, 1] data
DIVERGING_NEG = (0xE3, 0x49, 0x48)  # red: negative end
DIVERGING_MID = (0xF0, 0xEF, 0xEC)  # neutral gray midpoint
DIVERGING_POS = (0x2A, 0x78, 0xD6)  # blue: positive end (same blue as the first categorical colour)

# sequential single-hue ramp endpoints (blue, light -> dark) for magnitude data
SEQUENTIAL_LIGHT = (0xCD, 0xE2, 0xFB)  # light blue: low end
SEQUENTIAL_DARK = (0x0D, 0x36, 0x6B)  # dark blue: high end


def lerp(a, b, t):
    return a + (b - a) * t


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % tuple(max(0, min(255, round(c))) for c in rgb)


def make_diverging_color(neg_clamp=-1.0, pos_clamp=1.0, neg=DIVERGING_NEG, mid=DIVERGING_MID, pos=DIVERGING_POS):
    """Returns f(v) -> hex, gray at 0, ramping to `neg`/`pos` at neg_clamp/pos_clamp."""
    def f(v):
        if v <= 0:
            t = max(0.0, min(1.0, v / neg_clamp))
            rgb = tuple(lerp(mid[i], neg[i], t) for i in range(3))
        else:
            t = max(0.0, min(1.0, v / pos_clamp))
            rgb = tuple(lerp(mid[i], pos[i], t) for i in range(3))
        return rgb_to_hex(rgb)
    return f


def make_sequential_color(v_min, v_max, light=SEQUENTIAL_LIGHT, dark=SEQUENTIAL_DARK):
    """Returns f(v) -> hex, light at v_min, dark at v_max."""
    def f(v):
        t = max(0.0, min(1.0, (v - v_min) / (v_max - v_min))) if v_max > v_min else 0.0
        rgb = tuple(lerp(light[i], dark[i], t) for i in range(3))
        return rgb_to_hex(rgb)
    return f


def render_svg(svg_path, density=220, png=True, pdf=True):
    """Render an SVG to PNG/PDF alongside it, preferring rsvg-convert over ImageMagick.

    - ImageMagick 7's internal SVG renderer finds no usable font on macOS (`unable to read
      font ''`); it handles SVG properly only with librsvg as its delegate anyway.
    - `rsvg-convert -f pdf` writes a vector PDF; ImageMagick rasterizes first.

    The ImageMagick fallback is for systems without librsvg.
    """
    base = svg_path.rsplit(".svg", 1)[0]

    if shutil.which("rsvg-convert"):
        if png:
            subprocess.run(
                ["rsvg-convert", "--dpi-x", str(density), "--dpi-y", str(density),
                 "-b", "white", "-o", f"{base}.png", svg_path],
                check=True,
            )
        if pdf:
            subprocess.run(
                ["rsvg-convert", "-f", "pdf", "-b", "white", "-o", f"{base}.pdf", svg_path],
                check=True,
            )
        return

    im = shutil.which("magick") or shutil.which("convert")
    if im is None:
        raise RuntimeError(
            f"Cannot render {svg_path}: neither `rsvg-convert` nor ImageMagick "
            "(`magick`/`convert`) is on PATH. Install librsvg (brew install librsvg / "
            "apt-get install librsvg2-bin)."
        )
    im_args = [im, "convert"] if im.endswith("magick") else [im]
    if png:
        subprocess.run(
            im_args + ["-density", str(density), "-background", "white", svg_path, f"{base}.png"],
            check=True,
        )
    if pdf:
        subprocess.run(
            im_args + ["-background", "white", svg_path, f"{base}.pdf"],
            check=True,
        )


def load_trait_categories(path=None):
    """Return (trait_id -> category name, sorted list of category names, category -> hex color)."""
    if path is None:
        path = repo_path("data", "traits", "trait_categories.json")
    data = json.load(open(path))
    cat_map = {int(k): v for k, v in data["trait_categories"].items()}
    category_names = sorted(set(cat_map.values()))
    color_map = {name: CATEGORICAL_COLORS[i % 4] for i, name in enumerate(category_names)}
    return cat_map, category_names, color_map
