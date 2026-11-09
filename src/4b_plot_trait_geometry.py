"""Cosine-similarity heatmap of the trait probe directions.

Reads the geometry JSON written by src/4a_analyze_trait_geometry.py and writes
cosine_similarity_heatmap<suffix>.{svg,png,pdf} next to it (plain SVG, rendered to PNG/PDF by
utils.svg_plot_utils.render_svg).

    python -m src.4b_plot_trait_geometry \
        --results_json data/plots/probe_geometry/apertus_v1.5_8B/geometry_l1_0.0.json
"""
import json, os
import fire
import numpy as np

from src.utils.svg_plot_utils import (
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE,
    make_diverging_color, render_svg,
)


def plot_cosine_heatmap(d, out_dir, suffix):
    trait_ids = d["trait_ids"]
    trait_names = d["trait_names"]
    categories = d["categories"]
    mean_sim = np.array(d["mean_similarity_matrix"])
    n = len(trait_ids)

    off_diag = mean_sim[~np.eye(n, dtype=bool)]
    neg_clamp = min(-0.05, off_diag.min() * 1.1)  # a little past the observed min, at least -0.05
    color_fn = make_diverging_color(neg_clamp=neg_clamp, pos_clamp=1.0)

    CELL = 20
    PAD_L, PAD_T = 170, 96
    PAD_R, PAD_B = 96, 34
    W = PAD_L + n * CELL + PAD_R
    H = PAD_T + n * CELL + PAD_B

    # No title or subtitle: the figure caption carries that information.
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">']
    svg.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

    for i in range(n):
        y = PAD_T + i * CELL
        for j in range(n):
            x = PAD_L + j * CELL
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL-0.5}" height="{CELL-0.5}" fill="{color_fn(mean_sim[i][j])}"/>')

    for i, tid in enumerate(trait_ids):
        y = PAD_T + i * CELL + CELL - 5
        svg.append(f'<text x="{PAD_L-8}" y="{y:.1f}" font-size="9.5" fill="{TEXT_SECONDARY}" text-anchor="end">T{tid} {trait_names[i]}</text>')
    for j, tid in enumerate(trait_ids):
        x = PAD_L + j * CELL + CELL / 2
        svg.append(f'<text x="{x:.1f}" y="{PAD_T-8}" font-size="9" fill="{TEXT_SECONDARY}" text-anchor="start" transform="rotate(-60 {x:.1f} {PAD_T-8})">T{tid}</text>')

    prev_cat = None
    for i, cat in enumerate(categories):
        if prev_cat is not None and cat != prev_cat:
            y = PAD_T + i * CELL
            svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L+n*CELL:.1f}" y2="{y:.1f}" stroke="{TEXT_PRIMARY}" stroke-width="1.2" opacity="0.5"/>')
            svg.append(f'<line x1="{y-PAD_T+PAD_L:.1f}" y1="{PAD_T}" x2="{y-PAD_T+PAD_L:.1f}" y2="{PAD_T+n*CELL:.1f}" stroke="{TEXT_PRIMARY}" stroke-width="1.2" opacity="0.5"/>')
        prev_cat = cat

    cb_x = PAD_L + n * CELL + 24
    cb_y, cb_h, cb_w = PAD_T, n * CELL, 18
    steps = 60
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        v0 = neg_clamp + t0 * (1.0 - neg_clamp)
        v1 = neg_clamp + t1 * (1.0 - neg_clamp)
        y0 = cb_y + cb_h * (1 - t1)
        y1 = cb_y + cb_h * (1 - t0)
        svg.append(f'<rect x="{cb_x}" y="{y0:.1f}" width="{cb_w}" height="{(y1-y0)+0.5:.1f}" fill="{color_fn((v0+v1)/2)}"/>')
    svg.append(f'<text x="{cb_x+cb_w+6}" y="{cb_y+4}" font-size="11" fill="{TEXT_SECONDARY}">1.0</text>')
    zero_y = cb_y + cb_h * (1 - (0 - neg_clamp) / (1.0 - neg_clamp))
    svg.append(f'<text x="{cb_x+cb_w+6}" y="{zero_y+4:.1f}" font-size="11" fill="{TEXT_SECONDARY}">0.0</text>')
    svg.append(f'<text x="{cb_x+cb_w+6}" y="{cb_y+cb_h:.1f}" font-size="11" fill="{TEXT_SECONDARY}">{neg_clamp:.2f}</text>')
    lab_x, lab_y = cb_x + cb_w + 40, cb_y + cb_h / 2
    svg.append(f'<text x="{lab_x:.1f}" y="{lab_y:.1f}" font-size="12" fill="{TEXT_SECONDARY}" '
               f'text-anchor="middle" transform="rotate(-90 {lab_x:.1f} {lab_y:.1f})">cosine similarity</text>')
    svg.append("</svg>")

    out_path = os.path.join(out_dir, f"cosine_similarity_heatmap{suffix}.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf)")


def main(results_json: str, suffix: str = "", out_dir: str = None):
    d = json.load(open(results_json))
    # Default alongside the geometry json it came from, so plots stay with their checkpoint.
    out_dir = out_dir or os.path.dirname(os.path.abspath(results_json))
    os.makedirs(out_dir, exist_ok=True)
    plot_cosine_heatmap(d, out_dir, suffix)


if __name__ == "__main__":
    fire.Fire(main)
