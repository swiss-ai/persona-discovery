"""Probe cosine similarity against cross-trait behavioural transfer, one point per off-diagonal cell.

Reads data/plots/probe_geometry/<checkpoint>/cross_trait_matrix.json
(src/7a_build_cross_trait_matrix.py) and writes geometry_vs_transfer.{svg,png,pdf} next to it.

No regression line: cells that share a trait are not independent observations, and a fitted line
would suggest they are (the Mantel permutation tests in src/7c_analyze_cross_trait_correlation.py
account for the dependence).

    python -m src.8c_plot_geometry_vs_transfer
"""
import json, os
import fire
from src.utils.paths import repo_path

from src.utils.svg_plot_utils import (
    TEXT_SECONDARY, SURFACE, GRID, CATEGORICAL_COLORS, render_svg,
)



def plot_geometry_vs_transfer(m, out_dir, suffix=""):
    C, D = m["cosine_similarity"], m["d"]
    n = len(m["trait_ids"])
    pts = [(C[i][j], D[i][j]) for i in range(n) for j in range(n) if i != j]

    W, H = 620, 400
    PAD_L, PAD_R, PAD_T, PAD_B = 62, 16, 16, 52
    xs = [p[0] for p in pts]
    x_lo, x_hi = min(xs) - 0.015, max(xs) + 0.015
    y_lo, y_hi = -1.0, 1.0

    X = lambda v: PAD_L + (v - x_lo) / (x_hi - x_lo) * (W - PAD_L - PAD_R)
    Y = lambda v: PAD_T + (1 - (v - y_lo) / (y_hi - y_lo)) * (H - PAD_T - PAD_B)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']

    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        svg.append(f'<line x1="{PAD_L}" y1="{Y(t):.1f}" x2="{W-PAD_R}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{PAD_L-8}" y="{Y(t)+4:.1f}" font-size="11.5" fill="{TEXT_SECONDARY}" text-anchor="end">{t:+.1f}</text>')
    # d = 0: the steered answers are preferred equally often for the positive and the negative pole
    svg.append(f'<line x1="{PAD_L}" y1="{Y(0):.1f}" x2="{W-PAD_R}" y2="{Y(0):.1f}" '
               f'stroke="{TEXT_SECONDARY}" stroke-width="1.3" stroke-dasharray="5,4" opacity="0.65"/>')

    for xv in (0.0, 0.1, 0.2, 0.3):
        svg.append(f'<line x1="{X(xv):.1f}" y1="{Y(y_hi):.1f}" x2="{X(xv):.1f}" y2="{Y(y_lo):.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{X(xv):.1f}" y="{Y(y_lo)+18:.1f}" font-size="11.5" fill="{TEXT_SECONDARY}" text-anchor="middle">{xv:.1f}</text>')

    for x, y in pts:
        svg.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.4" fill="{CATEGORICAL_COLORS[0]}" fill-opacity="0.34"/>')

    svg.append(f'<line x1="{PAD_L}" y1="{Y(y_lo):.1f}" x2="{W-PAD_R}" y2="{Y(y_lo):.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1.2"/>')
    svg.append(f'<line x1="{PAD_L}" y1="{Y(y_hi):.1f}" x2="{PAD_L}" y2="{Y(y_lo):.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1.2"/>')
    svg.append(f'<text x="{PAD_L+(W-PAD_L-PAD_R)/2:.1f}" y="{H-12}" font-size="13" fill="{TEXT_SECONDARY}" text-anchor="middle">cosine similarity between trait directions</text>')
    yc = PAD_T + (H - PAD_T - PAD_B) / 2
    svg.append(f'<text x="17" y="{yc:.1f}" font-size="13" fill="{TEXT_SECONDARY}" text-anchor="middle" transform="rotate(-90 17 {yc:.1f})">pole contrast d</text>')
    svg.append("</svg>")

    out_path = os.path.join(out_dir, f"geometry_vs_transfer{suffix}.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf) — {len(pts)} cells")


def main(matrix_json: str = None, checkpoint: str = "apertus_v1.5_8B", out_dir: str = None):
    matrix_json = matrix_json or repo_path("data", "plots", "probe_geometry", checkpoint, "cross_trait_matrix.json")
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    plot_geometry_vs_transfer(json.load(open(matrix_json)), out_dir)


if __name__ == "__main__":
    fire.Fire(main)
