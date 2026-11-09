"""Trait x layer heatmap of probe ROC-AUC on the held-out activation split (src/3_train_probes.py).

Each cell is the ROC-AUC averaged over `l1_values`. Traits are sorted by the earliest layer at which
it reaches `threshold`, with a category-colour strip per row. The per-layer ROC-AUC is read from a cached pull, by default
data/plots/probe_geometry/<checkpoint>/layer_acc_l1_<l1>_cache.json (shipped for apertus_v1.5_8B,
l1 0.0 and 1.0), so the figure needs no W&B access. If the cache file does not exist (or
`--cache_json` names a new path), the values are pulled live from the W&B runs of
src.3_train_probes and written there; that needs the `wandb` and `pandas` packages, an API key at
`api_key_path`, and the environment variables WANDB_ENTITY / WANDB_PROJECT of those runs.

The default uses the most strongly penalised probes (l1=1.0), whose accuracy varies more across
layers than that of unpenalised probes.

    python -m src.4c_plot_layer_accuracy --checkpoint apertus_v1.5_8B --l1_values 1.0
"""
import os, json
import fire

from src.utils.svg_plot_utils import (
    TEXT_PRIMARY, TEXT_SECONDARY, SURFACE,
    make_sequential_color, render_svg, load_trait_categories,
)
from src.utils.paths import repo_path


def pull_layer_roc_auc(checkpoint, submodule, pooling, l1_values, n_traits, api_key_path):
    # Imported here, not at module scope: rendering from `cache_json` must work on a
    # machine without wandb installed and without a W&B key. Only a live pull needs it.
    import wandb

    try:
        entity, project = os.environ["WANDB_ENTITY"], os.environ["WANDB_PROJECT"]
    except KeyError as e:
        raise RuntimeError("A live W&B pull needs WANDB_ENTITY and WANDB_PROJECT, the entity and project "
                           "of the src.3_train_probes runs.") from e
    key = open(os.path.expanduser(api_key_path)).read().strip()
    os.environ["WANDB_API_KEY"] = key
    api = wandb.Api()
    cols = [f"test_roc_auc/{submodule}|{pooling}|l1={v}" for v in l1_values]

    results = {}
    for tid in range(1, n_traits + 1):
        runs = list(api.runs(f"{entity}/{project}",
                              filters={"config.trait_id": tid, "config.model.checkpoint": checkpoint},
                              per_page=20))
        if not runs:
            print(f"[WARN] trait {tid}: no W&B runs found for checkpoint={checkpoint}")
            continue
        run = sorted(runs, key=lambda r: r.created_at, reverse=True)[0]
        hist = run.history(samples=2000)
        if not all(c in hist.columns for c in cols) or "layer" not in hist.columns:
            print(f"[WARN] trait {tid}: run {run.id} missing expected columns")
            continue
        sub = hist[["layer"] + cols].dropna().sort_values("layer")
        layers = sub["layer"].astype(int).tolist()
        per_l1 = {v: sub[c].astype(float).tolist() for v, c in zip(l1_values, cols)}
        mean_vals = [sum(per_l1[v][i] for v in l1_values) / len(l1_values) for i in range(len(layers))]
        results[tid] = {"run_id": run.id, "layers": layers, "roc_auc_mean": mean_vals, "per_l1": per_l1}
        print(f"[INFO] trait {tid}: run {run.id}, layer0_mean={mean_vals[0]:.4f}, max_mean={max(mean_vals):.4f}")

    return results


def main(
    checkpoint: str = "apertus_v1.5_8B",
    submodule: str = "residual.post_mlp",
    pooling: str = "mean_pooling",
    l1_values=(1.0,),
    n_traits: int = 30,
    threshold: float = 0.90,
    api_key_path: str = "~/.wandb-api-key",
    categories_path: str = None,
    all_traits_path: str = None,
    out_dir: str = None,
    cache_json: str = None,
):
    categories_path = categories_path or repo_path("data", "traits", "trait_categories.json")
    all_traits_path = all_traits_path or repo_path("data", "traits", "all_traits.json")
    # One output folder per checkpoint.
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)

    l1_values = [str(v) for v in (l1_values if isinstance(l1_values, (list, tuple)) else [l1_values])]

    cache_json = cache_json or os.path.join(out_dir, f"layer_acc_l1_{'_'.join(l1_values)}_cache.json")
    if os.path.exists(cache_json):
        print(f"[INFO] loading cached pull from {cache_json}")
        layer_acc = json.load(open(cache_json))
        layer_acc = {int(k): v for k, v in layer_acc.items()}
    else:
        layer_acc = pull_layer_roc_auc(checkpoint, submodule, pooling, l1_values, n_traits, api_key_path)
        os.makedirs(os.path.dirname(os.path.abspath(cache_json)), exist_ok=True)
        json.dump(layer_acc, open(cache_json, "w"), indent=2)
        print(f"[INFO] cached pull to {cache_json}")

    cat_map, category_names, cat_color_map = load_trait_categories(categories_path)
    all_traits = json.load(open(all_traits_path))["therapist_traits"]
    name_map = {t["trait_id"]: t["trait_name_pos"] for t in all_traits}

    trait_ids = sorted(layer_acc.keys())
    n_layers = len(layer_acc[trait_ids[0]]["layers"])

    earliest = {}
    for tid in trait_ids:
        layers, vals = layer_acc[tid]["layers"], layer_acc[tid]["roc_auc_mean"]
        hit = next((L for L, v in zip(layers, vals) if v >= threshold), 999)
        earliest[tid] = hit
    order = sorted(trait_ids, key=lambda t: (earliest[t], t))

    all_vals = [v for tid in trait_ids for v in layer_acc[tid]["roc_auc_mean"]]
    color_fn = make_sequential_color(min(all_vals), max(all_vals))

    # Sizes chosen so the figure fits the text width with readable labels.
    CELL_W, ROW_H, CAT_STRIP_W, LABEL_W = 24, 23, 11, 190
    FS_TRAIT, FS_LAYER, FS_AXIS, FS_CBAR, FS_LEGEND = 13, 11.5, 13.5, 11.5, 12.5
    PAD_L = LABEL_W + CAT_STRIP_W + 8
    PAD_T = 26                      # layer numbers only
    PAD_R = 100                     # bar + tick labels + rotated axis label
    PAD_B = 54                      # x-axis label, then the category legend
    n = len(trait_ids)
    GRID_W = n_layers * CELL_W
    GRID_H = n * ROW_H
    W = PAD_L + GRID_W + PAD_R
    H = PAD_T + GRID_H + PAD_B

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">']
    svg.append(f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>')

    for i, tid in enumerate(order):
        y = PAD_T + i * ROW_H
        svg.append(f'<text x="{PAD_L - CAT_STRIP_W - 7}" y="{y+ROW_H-7:.1f}" font-size="{FS_TRAIT}" fill="{TEXT_PRIMARY}" text-anchor="end">T{tid} {name_map[tid]}</text>')
        svg.append(f'<rect x="{PAD_L - CAT_STRIP_W}" y="{y+1:.1f}" width="{CAT_STRIP_W-1}" height="{ROW_H-2}" fill="{cat_color_map[cat_map[tid]]}"/>')
        for L, v in zip(layer_acc[tid]["layers"], layer_acc[tid]["roc_auc_mean"]):
            x = PAD_L + L * CELL_W
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL_W-0.5}" height="{ROW_H-0.5}" fill="{color_fn(v)}"/>')
        if earliest[tid] != 999:
            x = PAD_L + earliest[tid] * CELL_W
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{CELL_W-0.5}" height="{ROW_H-0.5}" fill="none" stroke="{TEXT_PRIMARY}" stroke-width="1.3"/>')

    for L in range(n_layers):
        x = PAD_L + L * CELL_W + CELL_W / 2
        if L % 2 == 0:
            svg.append(f'<text x="{x:.1f}" y="{PAD_T-9}" font-size="{FS_LAYER}" fill="{TEXT_SECONDARY}" text-anchor="middle">{L}</text>')
    axis_y = PAD_T + GRID_H + 20
    svg.append(f'<text x="{PAD_L + GRID_W/2:.1f}" y="{axis_y}" font-size="{FS_AXIS}" fill="{TEXT_SECONDARY}" text-anchor="middle">layer</text>')

    # Category legend under the grid: swatch + label per category, laid out by a width estimate so
    # the row can be centred.
    SW, GAP_TL, GAP_ITEM = 11, 6, 20
    widths = [SW + GAP_TL + len(name) * FS_LEGEND * 0.54 for name in category_names]
    legend_w = sum(widths) + GAP_ITEM * (len(category_names) - 1)
    lx = PAD_L + GRID_W / 2 - legend_w / 2
    ly = axis_y + 24
    for name, w in zip(category_names, widths):
        svg.append(f'<rect x="{lx:.1f}" y="{ly-SW+1:.1f}" width="{SW}" height="{SW}" fill="{cat_color_map[name]}"/>')
        svg.append(f'<text x="{lx+SW+GAP_TL:.1f}" y="{ly}" font-size="{FS_LEGEND}" fill="{TEXT_PRIMARY}">{name}</text>')
        lx += w + GAP_ITEM

    cb_x, cb_y, cb_h, cb_w = PAD_L + GRID_W + 20, PAD_T, GRID_H, 18
    v_min, v_max = min(all_vals), max(all_vals)
    steps = 60
    for i in range(steps):
        t0, t1 = i / steps, (i + 1) / steps
        v0, v1 = v_min + t0 * (v_max - v_min), v_min + t1 * (v_max - v_min)
        y0, y1 = cb_y + cb_h * (1 - t1), cb_y + cb_h * (1 - t0)
        svg.append(f'<rect x="{cb_x}" y="{y0:.1f}" width="{cb_w}" height="{(y1-y0)+0.5:.1f}" fill="{color_fn((v0+v1)/2)}"/>')
    tick_x = cb_x + cb_w + 6
    svg.append(f'<text x="{tick_x}" y="{cb_y+9}" font-size="{FS_CBAR}" fill="{TEXT_SECONDARY}">{v_max:.2f}</text>')
    svg.append(f'<text x="{tick_x}" y="{cb_y+cb_h-2:.1f}" font-size="{FS_CBAR}" fill="{TEXT_SECONDARY}">{v_min:.2f}</text>')
    # Rotated so the label runs along the bar and is not clipped at the right edge.
    lab_x = tick_x + 34
    lab_y = cb_y + cb_h / 2
    svg.append(f'<text x="{lab_x:.1f}" y="{lab_y:.1f}" font-size="12" fill="{TEXT_SECONDARY}" '
               f'text-anchor="middle" transform="rotate(-90 {lab_x:.1f} {lab_y:.1f})">mean test ROC-AUC</text>')
    svg.append("</svg>")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "layer_probe_accuracy.svg")
    open(out_path, "w").write("\n".join(svg))
    render_svg(out_path)
    print(f"[INFO] wrote {out_path} (+ .png/.pdf)")


if __name__ == "__main__":
    fire.Fire(main)
