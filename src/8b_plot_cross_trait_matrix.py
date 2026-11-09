"""Cross-trait steering matrix, as two panels sharing a row/column order.

Reads data/plots/probe_geometry/<checkpoint>/cross_trait_matrix.json
(src/7a_build_cross_trait_matrix.py) and writes cross_trait_matrix.{svg,png,pdf} next to it.

  left   p, the steering preference rate when trait i's completions are judged for trait j
  right  d = p - n, the contrast between the positive pole (p) and the negative pole (n)

n itself is not shown; the negative-pole control is summarised in
src/7c_analyze_cross_trait_correlation.py.

Separate colour scales: p lies in [0, 1] and is centred on 0.5; d lies in [-1, 1] and is centred on
0. Both panels use the same diverging ramp about their own midpoint, with separate colourbars; the d
scale saturates at +-0.8. A shared scale would compress the d cells into the middle of the ramp.

    python -m src.8b_plot_cross_trait_matrix
"""
import json, os
import fire
from src.utils.paths import repo_path

from src.utils.svg_plot_utils import (
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE,
    make_diverging_color, render_svg,
)


CELL = 13
LABEL_EVERY = 5


def plot_cross_trait_matrix(d, out_dir, suffix=""):
    trait_ids = d["trait_ids"]
    categories = d["categories"]
    P, Dm = d["rate"], d["d"]
    n = len(trait_ids)

    # category boundaries, so the panels follow the order of the cosine heatmap
    bounds = [i for i in range(1, n) if categories[i] != categories[i - 1]]

    PAD_L, PAD_T, PAD_B, PAD_R = 34, 74, 30, 12
    CB_W, CB_GAP, PANEL_GAP = 15, 8, 56
    side = n * CELL
    cb_block = CB_GAP + CB_W + 40
    W = PAD_L + side + cb_block + PANEL_GAP + side + cb_block + PAD_R
    H = PAD_T + side + PAD_B

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">']
    svg.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

    def panel(x0, M, to_unit, title, ticks):
        color_fn = make_diverging_color(neg_clamp=-1.0, pos_clamp=1.0)
        svg.append(f'<text x="{x0}" y="{PAD_T-46}" font-size="14" font-weight="600" fill="{TEXT_PRIMARY}">{title}</text>')
        for i in range(n):
            for j in range(n):
                x, y = x0 + j * CELL, PAD_T + i * CELL
                svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL-0.4}" height="{CELL-0.4}" '
                           f'fill="{color_fn(to_unit(M[i][j]))}"/>')
        # Outline the diagonal (each trait judged for itself).
        for i in range(n):
            x, y = x0 + i * CELL, PAD_T + i * CELL
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL-0.4}" height="{CELL-0.4}" '
                       f'fill="none" stroke="{TEXT_PRIMARY}" stroke-width="1.1"/>')
        # tick labels every LABEL_EVERY traits, on both axes
        for i in range(n):
            if (i + 1) % LABEL_EVERY and i != 0:
                continue
            y = PAD_T + i * CELL + CELL - 3
            svg.append(f'<text x="{x0-5}" y="{y:.1f}" font-size="10" fill="{TEXT_SECONDARY}" text-anchor="end">T{trait_ids[i]}</text>')
            x = x0 + i * CELL + CELL / 2
            svg.append(f'<text x="{x:.1f}" y="{PAD_T-6}" font-size="10" fill="{TEXT_SECONDARY}" '
                       f'text-anchor="start" transform="rotate(-60 {x:.1f} {PAD_T-6})">T{trait_ids[i]}</text>')
        for b in bounds:
            o = b * CELL
            svg.append(f'<line x1="{x0}" y1="{PAD_T+o:.1f}" x2="{x0+side:.1f}" y2="{PAD_T+o:.1f}" stroke="{TEXT_PRIMARY}" stroke-width="1" opacity="0.45"/>')
            svg.append(f'<line x1="{x0+o:.1f}" y1="{PAD_T}" x2="{x0+o:.1f}" y2="{PAD_T+side:.1f}" stroke="{TEXT_PRIMARY}" stroke-width="1" opacity="0.45"/>')
        # colourbar, labelled in the panel's own units
        cx, cy = x0 + side + CB_GAP, PAD_T
        steps = 60
        for k in range(steps):
            t0, t1 = k / steps, (k + 1) / steps
            y0, y1 = cy + side * (1 - t1), cy + side * (1 - t0)
            svg.append(f'<rect x="{cx}" y="{y0:.1f}" width="{CB_W}" height="{(y1-y0)+0.5:.1f}" '
                       f'fill="{color_fn(-1 + 2 * (t0 + t1) / 2)}"/>')
        for frac, lab in ticks:
            ty = cy + side * (1 - frac)
            svg.append(f'<text x="{cx+CB_W+4}" y="{ty+3.5:.1f}" font-size="10" fill="{TEXT_SECONDARY}">{lab}</text>')

    # p in [0,1] about 0.5; d about 0, colour saturating at +-0.8
    panel(PAD_L, P, lambda v: (v - 0.5) / 0.5, "preference rate p",
          [(1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")])
    panel(PAD_L + side + cb_block + PANEL_GAP, Dm, lambda v: max(-1.0, min(1.0, v / 0.8)), "pole contrast d",
          [(1.0, "+0.8"), (0.5, "0.0"), (0.0, "\u22120.8")])

    svg.append("</svg>")
    out_path = os.path.join(out_dir, f"cross_trait_matrix{suffix}.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf)")


def main(results_json: str = None, checkpoint: str = "apertus_v1.5_8B", out_dir: str = None):
    results_json = results_json or repo_path("data", "plots", "probe_geometry", checkpoint, "cross_trait_matrix.json")
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    plot_cross_trait_matrix(json.load(open(results_json)), out_dir)


if __name__ == "__main__":
    fire.Fire(main)
