"""Angular (two-direction) steering: summary statistics recomputed from the per-pair rate vectors.

Reads data/plots/probe_geometry/<checkpoint>/angular_steering.json (src/7b_build_angular_steering.py),
recomputes every summary quantity from the raw per-pair rates, checks them against the stored values
(tolerance `tol`), and writes angular_summary_recomputed.json next to it.

Definitions:

  separability = min( max_theta(rate_a - rate_b), max_theta(rate_b - rate_a) )
      Can either trait be expressed while the other is suppressed? Taking the minimum of the two
      one-sided maxima means a pair only scores high if the separation works in both directions.
      A project-specific measure without a standard name.

  corr_ab = Pearson correlation between rate_a and rate_b across the mixing angles (12 in the
      reference experiment). Descriptive only: few angles per pair, bounded rates and pairs at the
      ceiling make it a description of co-movement, so no p-value is computed.

  Spearman correlations between mean cosine and separability are computed over all pairs, without
  pairs below cosine 0.05, and without pairs containing trait 11. They are exploratory: the pairs of
  the reference experiment (13) were selected among strong steerers rather than sampled.

    python -m src.7f_angular_summary
"""
import os, json, datetime
import numpy as np
import fire
from scipy.stats import pearsonr, spearmanr

from src.utils.paths import repo_path


def main(checkpoint: str = "apertus_v1.5_8B", out_dir: str = None, tol: float = 5e-3):
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    src = json.load(open(os.path.join(out_dir, "angular_steering.json")))
    pairs = src["pairs"]

    rows, mismatch = [], []
    for p in pairs:
        a, b = np.array(p["rate_a"], float), np.array(p["rate_b"], float)
        mab, mba = float(np.max(a - b)), float(np.max(b - a))
        sep = float(min(mab, mba))
        corr = float(pearsonr(a, b)[0])
        rec = {"trait_a": p["trait_a"], "trait_b": p["trait_b"],
               "cos_mean": p["cos_mean"], "n_angles": len(a),
               "max_a_minus_b": mab, "max_b_minus_a": mba,
               "separability_recomputed": sep, "separability_stored": p["separability"],
               "corr_ab_pearson_recomputed": corr, "corr_ab_stored": p["corr_ab"],
               "corr_ab_spearman_for_reference": float(spearmanr(a, b).correlation),
               "max_incoherence": float(np.max(p["incoherence"])),
               "involves_trait_11": p["involves_trait_11"]}
        for k1, k2 in [("separability_recomputed", "separability_stored"),
                       ("corr_ab_pearson_recomputed", "corr_ab_stored")]:
            if abs(rec[k1] - rec[k2]) > tol:
                mismatch.append((p["trait_a"], p["trait_b"], k1, rec[k1], rec[k2]))
        rows.append(rec)

    print(f"[INFO] {len(rows)} pairs; recomputed vs stored mismatches (tol {tol}): {len(mismatch)}")
    for m in mismatch:
        print("   MISMATCH", m)

    cos = np.array([r["cos_mean"] for r in rows])
    sep = np.array([r["separability_recomputed"] for r in rows])
    t11 = np.array([r["involves_trait_11"] for r in rows])
    stats = {}
    for name, mask in [("all", np.ones(len(rows), bool)),
                       ("excluding_cos_below_0.05", cos >= 0.05),
                       ("excluding_trait11_pairs", ~t11)]:
        r = spearmanr(cos[mask], sep[mask])
        stats[name] = {"rho": float(r.correlation), "p": float(r.pvalue), "n": int(mask.sum())}
        print(f"[INFO] Spearman(cos, separability) [{name}]: rho={r.correlation:+.3f} "
              f"p={r.pvalue:.4f} n={int(mask.sum())}")

    payload = {
        "_meta": {
            "description": "Angular steering summary statistics, recomputed from the per-pair rate "
                           "vectors in angular_steering.json.",
            "separability": "min(max(rate_a - rate_b), max(rate_b - rate_a)) over the 12 angles. "
                            "A project-specific measure without a standard name.",
            "corr_ab": "Pearson correlation of rate_a against rate_b across the 12 angles. Descriptive "
                       "only (n=12, bounded rates, ceiling effects); no p-value is computed.",
            "spearman_tests": "Exploratory: the 13 pairs were selected among strong steerers, not "
                              "sampled, and only two sit below cosine 0.05.",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "checkpoint": checkpoint, "n_pairs": len(rows), "tolerance": tol,
        "recomputation_mismatches": mismatch,
        "spearman_cos_vs_separability": stats,
        "pairs": rows,
    }
    os.makedirs(out_dir, exist_ok=True)
    op = os.path.join(out_dir, "angular_summary_recomputed.json")
    json.dump(payload, open(op, "w"), indent=2)
    print(f"\n[INFO] wrote {op}")


if __name__ == "__main__":
    fire.Fire(main)
