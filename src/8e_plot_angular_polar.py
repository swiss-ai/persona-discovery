"""The mixing sweep of the angular steering experiment in polar form, for two contrasting pairs.

Reads data/plots/probe_geometry/<checkpoint>/angular_steering.json
(src/7b_build_angular_steering.py) and writes angular_polar<suffix>.{svg,png,pdf} next to it.

Polar form because theta is cyclic: d(theta) = cos(theta)*unit(w_A) + sin(theta)*unit(w_B) returns to
itself after 360 degrees, and the four cardinal directions are +A, +B, -A, -B. A Cartesian axis would
cut the circle open at an arbitrary point.

The default `pairs` both contain Empathy (trait 7), so differences between the panels come from the
partner direction: (7,11) is the least similar pair of the reference experiment and (7,24) one of the
most similar. Coherence is not plotted; in the reference figure the degeneration rate was zero at all
plotted mixing values.

    python -m src.8e_plot_angular_polar
"""
import json, os, math
import fire
from src.utils.paths import repo_path
from src.utils.svg_plot_utils import (
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, GRID, CATEGORICAL_COLORS, render_svg,
)


PAIRS = [(7, 11), (7, 24)]


def plot_angular_polar(d, out_dir, pairs=PAIRS, suffix=""):
    angles = d["_meta"]["angles_deg"]
    by_id = {(p["trait_a"], p["trait_b"]): p for p in d["pairs"]}

    R = 108
    PANEL_W = 300
    PAD_T, PAD_B, PAD_L = 78, 74, 24   # room for the +B label above the circle, under the titles
    W, H = PAD_L + len(pairs) * PANEL_W, PAD_T + 2 * R + PAD_B

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']

    for k, key in enumerate(pairs):
        p = by_id[key]
        cx = PAD_L + k * PANEL_W + PANEL_W / 2 - 10
        cy = PAD_T + R

        def XY(theta_deg, rate):
            a = math.radians(theta_deg)
            r = R * max(0.0, min(1.0, rate))
            return cx + r * math.cos(a), cy - r * math.sin(a)

        for frac in (0.25, 0.5, 0.75, 1.0):
            dash = ' stroke-dasharray="4,4"' if frac == 0.5 else ""
            wdt = 1.3 if frac == 0.5 else 1
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R*frac:.1f}" fill="none" '
                       f'stroke="{TEXT_SECONDARY if frac==0.5 else GRID}" stroke-width="{wdt}"{dash} opacity="0.75"/>')
        for a in range(0, 360, 30):
            x2, y2 = XY(a, 1.0)
            svg.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{GRID}" stroke-width="1"/>')

        # the four cardinal mixing values are the pure poles of the two directions
        for a, lab in ((0, "+A"), (90, "+B"), (180, "−A"), (270, "−B")):
            lx, ly = XY(a, 1.0)
            ox, oy = (lx - cx) * 0.12, (ly - cy) * 0.12
            svg.append(f'<text x="{lx+ox:.1f}" y="{ly+oy+4:.1f}" font-size="11.5" fill="{TEXT_PRIMARY}" text-anchor="middle">{lab}</text>')
        # Radial scale: 0.5 and 1.0 on one spoke; 0 follows from them. Placed at 240 deg, where
        # the curves of the reference pairs are near zero, so the labels do not cover the data.
        for frac, lab in ((0.5, "0.5"), (1.0, "1.0")):
            tx, ty = XY(240, frac)
            # knock the ring out behind the text: the grid circle crosses the glyphs and
            # the leading stroke otherwise reads as a minus sign
            svg.append(f'<rect x="{tx-11:.1f}" y="{ty-14:.1f}" width="22" height="12" fill="{SURFACE}"/>')
            svg.append(f'<text x="{tx:.1f}" y="{ty-4:.1f}" font-size="9.5" fill="{TEXT_SECONDARY}" text-anchor="middle">{lab}</text>')

        for series, colour in ((p["rate_a"], CATEGORICAL_COLORS[0]), (p["rate_b"], CATEGORICAL_COLORS[1])):
            pts = [XY(a, v) for a, v in zip(angles, series)]
            path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
            r0, g0, b0 = int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)
            svg.append(f'<path d="{path}" fill="rgb({r0},{g0},{b0})" fill-opacity="0.13" stroke="{colour}" stroke-width="2.1" stroke-linejoin="round"/>')
            for x, y in pts:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.9" fill="{colour}" stroke="{SURFACE}" stroke-width="1"/>')

        cap = (f'cos {p["cos_mean"]:+.3f}, separability {p["separability"]:.2f}')
        svg.append(f'<text x="{cx:.1f}" y="{PAD_T-46:.1f}" font-size="12.5" font-weight="600" fill="{TEXT_PRIMARY}" text-anchor="middle">{p["name_a"]} and {p["name_b"]}</text>')
        svg.append(f'<text x="{cx:.1f}" y="{PAD_T-30:.1f}" font-size="11" fill="{TEXT_SECONDARY}" text-anchor="middle">{cap}</text>')

        ly0 = cy + R + 30
        for i, (nm, colour) in enumerate(((p["name_a"], CATEGORICAL_COLORS[0]), (p["name_b"], CATEGORICAL_COLORS[1]))):
            yy = ly0 + i * 16
            svg.append(f'<line x1="{cx-92:.1f}" y1="{yy}" x2="{cx-70:.1f}" y2="{yy}" stroke="{colour}" stroke-width="2.1"/>')
            svg.append(f'<circle cx="{cx-81:.1f}" cy="{yy}" r="2.9" fill="{colour}"/>')
            svg.append(f'<text x="{cx-63:.1f}" y="{yy+4}" font-size="11" fill="{TEXT_PRIMARY}">{nm}</text>')

    svg.append("</svg>")
    out_path = os.path.join(out_dir, f"angular_polar{suffix}.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf)")


def main(results_json: str = None, checkpoint: str = "apertus_v1.5_8B", out_dir: str = None,
         pairs=PAIRS, suffix: str = ""):
    """`pairs` selects which trait pairs are drawn, e.g. --pairs '[[7,11],[7,24]]'."""
    results_json = results_json or repo_path("data", "plots", "probe_geometry", checkpoint, "angular_steering.json")
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    pairs = [tuple(p) for p in pairs]
    plot_angular_polar(json.load(open(results_json)), out_dir, pairs=pairs, suffix=suffix)


if __name__ == "__main__":
    fire.Fire(main)
