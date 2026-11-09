"""Mean cosine similarity of a trait pair's probe directions against its separability, one point per pair.

Reads data/plots/probe_geometry/<checkpoint>/angular_steering.json
(src/7b_build_angular_steering.py) and writes angular_separability.{svg,png,pdf} next to it.

Separability is defined in src/7f_angular_summary.py. No fitted line: the pairs were selected, partly
on the predictor itself, so a line would imply an estimate the design cannot support. All points are
drawn alike, without highlighting particular traits.

    python -m src.8d_plot_angular_separability
"""
import json, os
import fire
from src.utils.paths import repo_path
from src.utils.svg_plot_utils import (
    TEXT_SECONDARY, SURFACE, GRID, CATEGORICAL_COLORS, render_svg,
)



def plot_angular_separability(d, out_dir, suffix=""):
    P = d["pairs"]
    W, H = 470, 300
    PAD_L, PAD_R, PAD_T, PAD_B = 56, 14, 14, 46
    x_lo, x_hi = -0.06, 0.43
    y_lo, y_hi = 0.0, 0.95
    X = lambda v: PAD_L + (v - x_lo) / (x_hi - x_lo) * (W - PAD_L - PAD_R)
    Y = lambda v: PAD_T + (1 - (v - y_lo) / (y_hi - y_lo)) * (H - PAD_T - PAD_B)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']

    for t in (0.0, 0.2, 0.4, 0.6, 0.8):
        svg.append(f'<line x1="{PAD_L}" y1="{Y(t):.1f}" x2="{W-PAD_R}" y2="{Y(t):.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{PAD_L-7}" y="{Y(t)+4:.1f}" font-size="11" fill="{TEXT_SECONDARY}" text-anchor="end">{t:.1f}</text>')
    for xv in (0.0, 0.1, 0.2, 0.3, 0.4):
        svg.append(f'<line x1="{X(xv):.1f}" y1="{PAD_T}" x2="{X(xv):.1f}" y2="{Y(y_lo):.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{X(xv):.1f}" y="{Y(y_lo)+17:.1f}" font-size="11" fill="{TEXT_SECONDARY}" text-anchor="middle">{xv:.1f}</text>')

    svg.append(f'<line x1="{PAD_L}" y1="{Y(y_lo):.1f}" x2="{W-PAD_R}" y2="{Y(y_lo):.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1.2"/>')
    svg.append(f'<line x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{Y(y_lo):.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1.2"/>')

    for p in P:
        x, y = X(p["cos_mean"]), Y(p["separability"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{CATEGORICAL_COLORS[0]}" fill-opacity="0.8"/>')

    svg.append(f'<text x="{PAD_L+(W-PAD_L-PAD_R)/2:.1f}" y="{H-10}" font-size="12.5" fill="{TEXT_SECONDARY}" text-anchor="middle">pairwise cosine similarity</text>')
    yc = PAD_T + (H - PAD_T - PAD_B) / 2
    svg.append(f'<text x="16" y="{yc:.1f}" font-size="12.5" fill="{TEXT_SECONDARY}" text-anchor="middle" transform="rotate(-90 16 {yc:.1f})">separability</text>')
    svg.append("</svg>")

    out_path = os.path.join(out_dir, f"angular_separability{suffix}.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf) — {len(P)} pairs")


def main(results_json: str = None, checkpoint: str = "apertus_v1.5_8B", out_dir: str = None):
    results_json = results_json or repo_path("data", "plots", "probe_geometry", checkpoint, "angular_steering.json")
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    plot_angular_separability(json.load(open(results_json)), out_dir)


if __name__ == "__main__":
    fire.Fire(main)
