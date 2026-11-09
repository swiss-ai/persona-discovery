"""Assemble the angular (two-direction) steering results, one record per trait pair.

For each pair (A, B), src/5b_steer_angular.py steered along d(theta) = cos(theta)*unit(w_A) +
sin(theta)*unit(w_B) at 12 mixing values theta. Reads, per pair directory pair_<A>_<B>:

  $SCRATCH/apertus-pulse/data/steering_angular/<checkpoint>/l1_<l1>/pair_<A>_<B>/meta.json
      (5b: trait names, angles, strength, per-layer cosine between the two probe directions)
  $SCRATCH/apertus-pulse/data/steering_angular_eval/.../pair_<A>_<B>/eval_trait_{A,B}/summary.json
      (src/6b_evaluate_steering.py: steering preference rate for A and for B at each theta)
  $SCRATCH/apertus-pulse/data/steering_angular_coherence/.../pair_<A>_<B>/summary.json
      (src/6a_score_coherence.py: absolute degeneration rate at each theta)

and writes data/plots/probe_geometry/<checkpoint>/angular_steering.json, pairs sorted by mean cosine.

  separability = min( max_theta(rate_a - rate_b), max_theta(rate_b - rate_a) )
  corr_ab      = Pearson correlation of rate_a and rate_b across the angles (descriptive only)

    python -m src.7b_build_angular_steering --checkpoint apertus_v1.5_8B
"""
import os, json, math, datetime
import numpy as np
import fire
from scipy.stats import pearsonr, spearmanr

from src.utils.paths import repo_path, data_path

# In the reference experiment trait 11 appears in both low-cosine pairs, so the summary statistics
# are also reported without the pairs that contain it.
TRAIT_11 = 11


def _pair(a, b, angles, meta, eval_a, eval_b, coherence):
    keys = [str(t) for t in angles]
    cos = list(meta["cos_by_layer"].values())
    cos_mean = float(np.mean(cos))
    rate_a = [eval_a[k]["steered_preferred_rate"] for k in keys]
    rate_b = [eval_b[k]["steered_preferred_rate"] for k in keys]
    ra, rb = np.array(rate_a, float), np.array(rate_b, float)
    max_ab, max_ba = float(np.max(ra - rb)), float(np.max(rb - ra))
    return {
        "trait_a": a, "trait_b": b, "name_a": meta["name_a"], "name_b": meta["name_b"],
        "cos_mean": cos_mean, "cos_min": float(np.min(cos)), "cos_max": float(np.max(cos)),
        "true_arc_deg": math.degrees(math.acos(cos_mean)),
        "cos_by_layer": meta["cos_by_layer"],
        "rate_a": rate_a, "rate_b": rate_b,
        "incoherence": [coherence[k]["incoherent_rate_absolute"] for k in keys],
        "corr_ab": float(pearsonr(ra, rb)[0]),
        "max_a_minus_b": max_ab, "max_b_minus_a": max_ba,
        "separability": min(max_ab, max_ba),
        "involves_trait_11": TRAIT_11 in (a, b),
    }


def main(checkpoint: str = "apertus_v1.5_8B", l1: str = "0.0", out_dir: str = None):
    l1 = str(l1)
    sub = os.path.join(checkpoint, f"l1_{l1}")
    steer_root = data_path("steering_angular", sub)
    eval_root = data_path("steering_angular_eval", sub)
    coh_root = data_path("steering_angular_coherence", sub)
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)

    pairs, run_meta = [], None
    for name in sorted(os.listdir(steer_root)):
        if not name.startswith("pair_"):
            continue
        meta = json.load(open(os.path.join(steer_root, name, "meta.json")))
        a, b = meta["trait_a"], meta["trait_b"]
        common = {k: meta[k] for k in ("checkpoint", "l1", "strength", "num_questions", "angles")}
        if run_meta is None:
            run_meta = common
        elif common != run_meta:
            raise ValueError(f"{name}: run settings differ from the other pairs: {common} vs {run_meta}")
        eval_a = json.load(open(os.path.join(eval_root, name, f"eval_trait_{a}", "summary.json")))
        eval_b = json.load(open(os.path.join(eval_root, name, f"eval_trait_{b}", "summary.json")))
        coherence = json.load(open(os.path.join(coh_root, name, "summary.json")))
        pairs.append(_pair(a, b, meta["angles"], meta, eval_a, eval_b, coherence))
    pairs.sort(key=lambda p: p["cos_mean"])

    cos = np.array([p["cos_mean"] for p in pairs])
    sep = np.array([p["separability"] for p in pairs])
    t11 = np.array([p["involves_trait_11"] for p in pairs])
    stats = {}
    for key, mask in [("spearman_cos_vs_separability_all", np.ones(len(pairs), bool)),
                      ("spearman_excluding_cos_below_0.05", cos >= 0.05),
                      ("spearman_excluding_trait11_pairs", ~t11)]:
        r = spearmanr(cos[mask], sep[mask])
        stats[key] = [float(r.correlation), float(r.pvalue)]

    payload = {
        "_meta": {
            "description": f"Angular / combined steering, {len(pairs)} pairs x {len(run_meta['angles'])} angles.",
            "checkpoint": run_meta["checkpoint"],
            "l1": run_meta["l1"],
            "strength": run_meta["strength"],
            "num_questions": run_meta["num_questions"],
            "angles_deg": run_meta["angles"],
            "theta_semantics": "theta=0 is exactly pure A and theta=90 exactly pure B at every layer. theta is "
                               "a mixing parameter, not a true angle: the real A->B arc is arccos(cosine) "
                               "and varies by layer.",
            "separability": "min(max(A-B), max(B-A)) over angles: can either trait be expressed while the "
                            "other is suppressed?",
            "stats": stats,
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "pairs": pairs,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "angular_steering.json")
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"[INFO] wrote {out_path} ({len(pairs)} pairs)")


if __name__ == "__main__":
    fire.Fire(main)
