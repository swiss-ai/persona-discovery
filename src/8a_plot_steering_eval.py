"""Steering preference rate and degeneration rate vs. steering strength, per checkpoint.

Reads the judge results of src.6b_evaluate_steering
($SCRATCH/apertus-pulse/data/steering_eval/<checkpoint>/l1_<l1>/trait_<id>/eval_trait_<id>/summary.json) and the
coherence scores of src.6a_score_coherence (steering_coherence/.../summary.json), and writes
data/plots/steering_eval/steering_spr_coherence.{svg,png,pdf}.

    python -m src.8a_plot_steering_eval
"""
import json, os, math
import fire
from src.utils.paths import repo_path, data_path

from src.utils.svg_plot_utils import (
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE, GRID,
    hex_to_rgb, render_svg,
)

DEFAULT_CHECKPOINTS = ["apertus_mini_5B_base", "apertus_mini_5B_instruct", "apertus_v1.5_8B"]
# Legend labels: the model names without the "Apertus" prefix. Apertus v1.5-8B is
# instruction-tuned as released (there is no separate -Instruct variant for v1.5, unlike v1.1).
DEFAULT_LABELS = {
    "apertus_mini_5B_base": "v1.1-0.5B",
    "apertus_mini_5B_instruct": "v1.1-0.5B-Instruct",
    "apertus_v1.5_8B": "v1.5-8B",
}
DEFAULT_COLORS = {
    "apertus_mini_5B_base": "#2a78d6",
    "apertus_mini_5B_instruct": "#eb6834",
    "apertus_v1.5_8B": "#1baf7a",
}


def _load_per_trait_rates(root, checkpoints, n_traits, l1="0.0"):
    """Per checkpoint: {strength: [steered-preferred rate of each trait]}."""
    per_ckpt_strength_rates = {c: {} for c in checkpoints}
    for ckpt in checkpoints:
        for tid in range(1, n_traits + 1):
            summary_path = os.path.join(root, ckpt, f"l1_{l1}", f"trait_{tid}", f"eval_trait_{tid}", "summary.json")
            if not os.path.exists(summary_path):
                continue
            d = json.load(open(summary_path))
            for row in d.values():
                s, rate = row["strength"], row["steered_preferred_rate"]
                if rate is not None:
                    per_ckpt_strength_rates[ckpt].setdefault(s, []).append(rate)
    return per_ckpt_strength_rates


def _load_coherence(root, checkpoints, n_traits, l1="0.0"):
    """Mean over traits of each trait's own absolute/relative failure fraction, per strength."""
    out = {c: {} for c in checkpoints}
    for ckpt in checkpoints:
        agg = {}
        for tid in range(1, n_traits + 1):
            p = os.path.join(root, ckpt, f"l1_{l1}", f"trait_{tid}", "summary.json")
            if not os.path.exists(p):
                continue
            for key, row in json.load(open(p)).items():
                if key == "_meta" or not isinstance(row, dict) or "strength" not in row:
                    continue
                s = row["strength"]
                agg.setdefault(s, {"abs": [], "rel": []})
                agg[s]["abs"].append(row["incoherent_rate_absolute"])
                agg[s]["rel"].append(row["incoherent_rate_relative"])
        out[ckpt] = {s: (sum(v["abs"]) / len(v["abs"]), sum(v["rel"]) / len(v["rel"]))
                     for s, v in agg.items()}
    return out


def plot_spr_with_coherence(per_ckpt_strength_rates, coherence, checkpoints, labels, colors, out_dir):
    """
    Two panels on a shared ordinal x-axis:
      top    steering preference rate vs strength, one line per checkpoint, 95% CI band
      bottom degeneration rate vs strength -- absolute (solid) and relative (dashed),
             both for every checkpoint

    The SPR at s = 0 is not judged: the baseline would be compared with itself. By convention the
    curve passes through 0.5 there, with a zero-width band, so each checkpoint is drawn as one
    continuous curve.

    The panels share the x-axis so that steering preference and degeneration can be read at the
    same strength. The axis is ordinal, not linear in s: every tested strength gets the same spacing,
    because the thesis strengths (-0.05 to +0.1) are concentrated within +/-0.02.
    """
    # ordinal x positions over the union of every tested strength, baseline included
    all_s = sorted(set(s for c in checkpoints for s in per_ckpt_strength_rates[c])
                   | set(s for c in checkpoints for s in coherence.get(c, {})))
    xi = {s: i for i, s in enumerate(all_s)}

    # Canvas width chosen so the labels stay readable when the figure is scaled to the text width.
    W = 780
    PAD_L, PAD_R, PAD_T, PAD_B = 66, 20, 14, 90
    H_TOP, H_BOT, GAP_Y = 235, 140, 58
    H = PAD_T + H_TOP + GAP_Y + H_BOT + PAD_B
    plot_w = W - PAD_L - PAD_R
    step = plot_w / (len(all_s) - 1)

    def xs(s):
        return PAD_L + xi[s] * step

    def y_top(pct):
        return PAD_T + (1 - pct / 100) * H_TOP

    def y_bot(pct):
        return PAD_T + H_TOP + GAP_Y + (1 - pct / 100) * H_BOT

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">']
    svg.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

    # ---- gridlines, axes, tick labels -------------------------------------------------
    for pct in [0, 25, 50, 75, 100]:
        y = y_top(pct)
        svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{PAD_L-10}" y="{y+4:.1f}" font-size="12" fill="{TEXT_SECONDARY}" text-anchor="end">{pct/100:.2f}</text>')
    for pct in [0, 50, 100]:
        y = y_bot(pct)
        svg.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        svg.append(f'<text x="{PAD_L-10}" y="{y+4:.1f}" font-size="12" fill="{TEXT_SECONDARY}" text-anchor="end">{pct/100:.2f}</text>')

    # chance reference on the top panel
    y50 = y_top(50)
    svg.append(f'<line x1="{PAD_L}" y1="{y50:.1f}" x2="{W-PAD_R}" y2="{y50:.1f}" '
               f'stroke="{TEXT_SECONDARY}" stroke-width="1.4" stroke-dasharray="5,4" opacity="0.6"/>')

    # the s = 0 column, marked in both panels
    x0 = xs(0.0)
    svg.append(f'<line x1="{x0:.1f}" y1="{PAD_T}" x2="{x0:.1f}" y2="{PAD_T+H_TOP}" stroke="{GRID}" stroke-width="1.5"/>')
    svg.append(f'<line x1="{x0:.1f}" y1="{y_bot(100):.1f}" x2="{x0:.1f}" y2="{y_bot(0):.1f}" stroke="{GRID}" stroke-width="1.5"/>')

    for pane_y0, pane_y1 in [(PAD_T, PAD_T + H_TOP), (y_bot(100), y_bot(0))]:
        svg.append(f'<line x1="{PAD_L}" y1="{pane_y0:.1f}" x2="{PAD_L}" y2="{pane_y1:.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1.2"/>')
        svg.append(f'<line x1="{PAD_L}" y1="{pane_y1:.1f}" x2="{W-PAD_R}" y2="{pane_y1:.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1.2"/>')

    y_axis_bot = y_bot(0)
    for s in all_s:
        x = xs(s)
        svg.append(f'<line x1="{x:.1f}" y1="{y_axis_bot:.1f}" x2="{x:.1f}" y2="{y_axis_bot+5:.1f}" stroke="{TEXT_SECONDARY}" stroke-width="1"/>')
        lab = "0" if s == 0 else f"{s:g}"
        svg.append(f'<text x="{x:.1f}" y="{y_axis_bot+20:.1f}" font-size="12" fill="{TEXT_SECONDARY}" '
                   f'text-anchor="end" transform="rotate(-45 {x:.1f} {y_axis_bot+20:.1f})">{lab}</text>')

    svg.append(f'<text x="{PAD_L + plot_w/2:.1f}" y="{H-30}" font-size="13" fill="{TEXT_SECONDARY}" text-anchor="middle">steering strength</text>')
    ytc = PAD_T + H_TOP / 2
    svg.append(f'<text x="18" y="{ytc:.1f}" font-size="13" fill="{TEXT_SECONDARY}" text-anchor="middle" '
               f'transform="rotate(-90 18 {ytc:.1f})">steering preference rate</text>')
    ybc = PAD_T + H_TOP + GAP_Y + H_BOT / 2
    svg.append(f'<text x="18" y="{ybc:.1f}" font-size="13" fill="{TEXT_SECONDARY}" text-anchor="middle" '
               f'transform="rotate(-90 18 {ybc:.1f})">degeneration rate</text>')

    # ---- top panel: SPR, one continuous curve per checkpoint through s = 0 -------------
    for ckpt in checkpoints:
        pts = []
        for s in sorted(per_ckpt_strength_rates[ckpt]):
            rates = [r * 100 for r in per_ckpt_strength_rates[ckpt][s]]
            n = len(rates)
            mean = sum(rates) / n
            var = sum((r - mean) ** 2 for r in rates) / (n - 1) if n > 1 else 0.0
            sem = math.sqrt(var / n) if n > 0 else 0.0
            pts.append((s, mean, max(0.0, mean - 1.96 * sem), min(100.0, mean + 1.96 * sem)))
        pts.append((0.0, 50.0, 50.0, 50.0))  # plotting convention for s = 0; see the docstring
        pts.sort(key=lambda t: t[0])
        r, g, b = hex_to_rgb(colors[ckpt])
        up = [(xs(s), y_top(hi)) for s, _, _, hi in pts]
        lo = [(xs(s), y_top(l)) for s, _, l, _ in pts][::-1]
        path = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in up + lo) + " Z"
        svg.append(f'<path d="{path}" fill="rgb({r},{g},{b})" fill-opacity="0.15" stroke="none"/>')
        d = " ".join(f'{"M" if i==0 else "L"}{xs(s):.1f},{y_top(v):.1f}' for i, (s, v, _, _) in enumerate(pts))
        svg.append(f'<path d="{d}" fill="none" stroke="{colors[ckpt]}" stroke-width="2.5" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
        for s, v, _, _ in pts:
            svg.append(f'<circle cx="{xs(s):.1f}" cy="{y_top(v):.1f}" r="4.5" fill="{colors[ckpt]}" stroke="{SURFACE}" stroke-width="1.5"/>')

    # ---- bottom panel: absolute (solid) and relative (dashed) degeneration rate -------
    for ckpt in checkpoints:
        series = coherence.get(ckpt, {})
        if not series:
            continue
        ss = sorted(series)
        d = " ".join(f'{"M" if i==0 else "L"}{xs(s):.1f},{y_bot(series[s][0]*100):.1f}' for i, s in enumerate(ss))
        svg.append(f'<path d="{d}" fill="none" stroke="{colors[ckpt]}" stroke-width="2.5" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
        for s in ss:
            svg.append(f'<circle cx="{xs(s):.1f}" cy="{y_bot(series[s][0]*100):.1f}" r="4" fill="{colors[ckpt]}" stroke="{SURFACE}" stroke-width="1.5"/>')

    for ckpt in checkpoints:
        series = coherence.get(ckpt, {})
        if not series:
            continue
        ss = sorted(series)
        d = " ".join(f'{"M" if i==0 else "L"}{xs(s):.1f},{y_bot(series[s][1]*100):.1f}' for i, s in enumerate(ss))
        svg.append(f'<path d="{d}" fill="none" stroke="{colors[ckpt]}" stroke-width="2.2" '
                   f'stroke-dasharray="6,4" stroke-linecap="round" stroke-linejoin="round"/>')

    # ---- legend, below the plot and centred -------------------------------------------
    # Colour carries the checkpoint; in the lower panel the stroke carries the criterion,
    # so the two style keys are drawn in neutral grey rather than repeating every colour.
    entries = [(labels[c], colors[c], False) for c in checkpoints]
    entries.append(("absolute", TEXT_SECONDARY, False))
    entries.append(("relative", TEXT_SECONDARY, True))
    widths = [34 + len(t) * 6.6 + 22 for t, _, _ in entries]
    lx = PAD_L + (plot_w - sum(widths)) / 2
    ly = H - 12
    for (text, color, dashed), wdt in zip(entries, widths):
        dash = ' stroke-dasharray="6,4"' if dashed else ""
        svg.append(f'<line x1="{lx:.1f}" y1="{ly}" x2="{lx+24:.1f}" y2="{ly}" stroke="{color}" stroke-width="2.5" stroke-linecap="round"{dash}/>')
        if not dashed:
            svg.append(f'<circle cx="{lx+12:.1f}" cy="{ly}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{lx+32:.1f}" y="{ly+4}" font-size="12.5" fill="{TEXT_PRIMARY}">{text}</text>')
        lx += wdt
    svg.append("</svg>")

    out_path = os.path.join(out_dir, "steering_spr_coherence.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf)")


def main(
    steering_eval_dir: str = None,
    coherence_dir: str = None,
    l1: str = "0.0",
    checkpoints=DEFAULT_CHECKPOINTS,
    n_traits: int = 30,
    out_dir: str = None,
):
    steering_eval_dir = steering_eval_dir or data_path("steering_eval")
    out_dir = out_dir or repo_path("data", "plots", "steering_eval")
    checkpoints = list(checkpoints)
    labels = {c: DEFAULT_LABELS.get(c, c) for c in checkpoints}
    colors = {c: DEFAULT_COLORS.get(c, "#2a78d6") for c in checkpoints}

    per_ckpt_strength_rates = _load_per_trait_rates(steering_eval_dir, checkpoints, n_traits, l1)
    coherence = _load_coherence(coherence_dir or data_path("steering_coherence"), checkpoints, n_traits, l1)

    os.makedirs(out_dir, exist_ok=True)
    plot_spr_with_coherence(per_ckpt_strength_rates, coherence, checkpoints, labels, colors, out_dir)


if __name__ == "__main__":
    fire.Fire(main)
