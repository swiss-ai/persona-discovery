"""Scope check for the ARI between probe-geometry clusters and the expert trait categories.

The categories in data/traits/trait_categories.json are unbalanced (2 / 13 / 6 / 9 traits), and
k-means with k=4 tends toward comparably sized clusters, so it may be unable to recover a small
category whatever the geometry contains. This script quantifies that limit; it does not search for a
clustering method that recovers the taxonomy.

Reads data/plots/probe_geometry/<checkpoint>/geometry_l1_<l1>.json (src/4a_analyze_trait_geometry.py)
for the three checkpoints and five L1 values (files that do not exist are skipped), and writes
data/plots/probe_geometry/clustering_scope_check.json. Three checks:

1. The cluster-size profiles that k-means and hierarchical clustering actually produced.
2. An estimate of the highest ARI that any partition with the produced cluster sizes can reach against
   the category sizes, from a hill-climbing search with random restarts. It is the highest value found,
   not a proven maximum; a value well below 1 shows that the cluster sizes alone limit the ARI.
3. A permutation null that keeps the category sizes: the fitted cluster labels stay fixed and the
   category labels are shuffled. It tests whether an observed ARI exceeds chance given the cluster
   sizes the method produced.

    python -m src.4d_clustering_scope_check
"""
import os, json, datetime, collections
import numpy as np
import fire
from sklearn.metrics import adjusted_rand_score

from src.utils.paths import repo_path

N_RESTARTS = 60  # random starts of the ARI-ceiling hill climb

CKPTS = ["apertus_v1.5_8B", "apertus_mini_5B_base", "apertus_mini_5B_instruct"]
L1S = ["0.0", "0.01", "0.1", "0.5", "1.0"]


def _ari_from_table(T, n):
    """ARI computed directly from a contingency table -- ARI depends on nothing else."""
    c2 = lambda x: x * (x - 1) / 2.0
    idx = c2(T).sum()
    a = c2(T.sum(1)).sum()
    b = c2(T.sum(0)).sum()
    exp = a * b / c2(n)
    mx = (a + b) / 2.0
    return 0.0 if mx == exp else float((idx - exp) / (mx - exp))


def _ari_ceiling(cluster_sizes, cat_sizes, n_restarts=N_RESTARTS, seed=0):
    """Highest ARI found for a partition with these cluster sizes against these category sizes.

    Hill climbing from `n_restarts` random starts; the result is a lower bound on the true maximum.
    ARI depends only on the contingency table, so the search runs over integer tables with the given
    margins instead of over label vectors (the same set of reachable ARI values, much cheaper).
    """
    rng = np.random.default_rng(seed)
    r = np.array(cluster_sizes, float); c = np.array(cat_sizes, float)
    n = int(r.sum()); best = -1.0
    for t in range(n_restarts):
        # Seed with a random transportation plan, then greedily move mass to raise ARI.
        T = np.zeros((len(r), len(c)))
        rr, cc = r.copy(), c.copy()
        order = rng.permutation(len(r))
        for i in order:
            for j in rng.permutation(len(c)):
                take = min(rr[i], cc[j])
                T[i, j] += take; rr[i] -= take; cc[j] -= take
        cur = _ari_from_table(T, n)
        improved = True
        while improved:
            improved = False
            for i1 in range(len(r)):
                for i2 in range(len(r)):
                    for j1 in range(len(c)):
                        for j2 in range(len(c)):
                            if i1 == i2 or j1 == j2 or T[i1, j1] < 1 or T[i2, j2] < 1:
                                continue
                            T[i1, j1] -= 1; T[i2, j2] -= 1; T[i1, j2] += 1; T[i2, j1] += 1
                            s2 = _ari_from_table(T, n)
                            if s2 > cur + 1e-12:
                                cur, improved = s2, True
                            else:
                                T[i1, j1] += 1; T[i2, j2] += 1; T[i1, j2] -= 1; T[i2, j1] -= 1
        best = max(best, cur)
    return float(best)


def _ari_perm_null(cluster_labels, cat_labels, n_perm=20000, seed=0):
    """Null preserving true category sizes: shuffle categories, keep the fitted clusters fixed."""
    rng = np.random.default_rng(seed)
    obs = adjusted_rand_score(cat_labels, cluster_labels)
    cats = np.array(cat_labels)
    null = np.array([adjusted_rand_score(rng.permutation(cats), cluster_labels) for _ in range(n_perm)])
    # (b+1)/(m+1), the same estimator as the other permutation tests (Phipson & Smyth 2010).
    # One-sided: the hypothesis is directional (agreement higher than chance).
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return float(obs), float(null.mean()), float(null.std()), p


def main(n_perm: int = 20000, seed: int = 0, out_dir: str = None):
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry")
    rows, sizes_seen = [], {"kmeans": collections.Counter(), "agglo": collections.Counter()}
    ceilings = {}

    for ck in CKPTS:
        for l1 in L1S:
            p = os.path.join(out_dir, ck, f"geometry_l1_{l1}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            cats = d["categories"]
            cat_names = sorted(set(cats))
            cat_idx = [cat_names.index(c) for c in cats]
            for method in ("kmeans", "agglo"):
                lab = d[f"{method}_labels"]
                sz = tuple(sorted(collections.Counter(lab).values(), reverse=True))
                sizes_seen[method][sz] += 1
                obs, nm, ns, pv = _ari_perm_null(lab, cat_idx, n_perm, seed)
                csz = tuple(sorted(collections.Counter(cat_idx).values(), reverse=True))
                key = (sz, csz)
                if key not in ceilings:
                    ceilings[key] = _ari_ceiling(sz, csz, seed=seed)
                rows.append({"checkpoint": ck, "l1": l1, "method": method,
                             "cluster_sizes": list(sz), "ari": obs,
                             "ari_null_mean": nm, "ari_null_sd": ns, "ari_perm_p": pv,
                             "ari_ceiling_for_these_sizes": ceilings[key]})

    true_sizes = tuple(sorted(collections.Counter(
        json.load(open(os.path.join(out_dir, CKPTS[0], "geometry_l1_0.0.json")))["categories"]).values(),
        reverse=True))
    print(f"True category sizes: {true_sizes}\n")

    print("Cluster-size profiles actually produced (count over 15 runs each):")
    for method in ("kmeans", "agglo"):
        print(f"  {method}:")
        for sz, c in sizes_seen[method].most_common():
            flag = "  <-- matches true profile" if sz == true_sizes else ""
            print(f"    {str(list(sz)):<18} x{c}{flag}")
    kmin = [min(r["cluster_sizes"]) for r in rows if r["method"] == "kmeans"]
    amin = [min(r["cluster_sizes"]) for r in rows if r["method"] == "agglo"]
    print(f"\n  smallest cluster ever produced: kmeans={min(kmin)}, agglo={min(amin)} "
          f"(a 2-member category exists in the truth)")

    print(f"\n{'checkpoint':<26}{'l1':>5}{'method':>9}{'ARI':>8}{'null':>8}{'p':>8}{'ceiling':>9}  sizes")
    for r in sorted(rows, key=lambda r: -r["ari"]):
        print(f"{r['checkpoint']:<26}{r['l1']:>5}{r['method']:>9}{r['ari']:>8.3f}"
              f"{r['ari_null_mean']:>8.3f}{r['ari_perm_p']:>8.4f}"
              f"{r['ari_ceiling_for_these_sizes']:>9.3f}  {r['cluster_sizes']}")

    sig = [r for r in rows if r["ari_perm_p"] < 0.05]
    print(f"\nRuns with ARI above a size-preserving permutation null at p<0.05: {len(sig)}/{len(rows)}")
    for r in sig:
        print(f"  {r['checkpoint']} l1={r['l1']} {r['method']}: ARI={r['ari']:.3f}, p={r['ari_perm_p']:.4f}")

    payload = {
        "_meta": {
            "description": "Scope check for the ARI-based taxonomy comparison.",
            "purpose": "Bound what the near-zero ARI can support, given severely unbalanced category "
                       "sizes (2/13/6/9). NOT an attempt to raise the ARI.",
            "search_maximum": "Highest ARI FOUND for any partition with the cluster sizes the method "
                       f"actually produced, by hill-climbing from {N_RESTARTS} random starts. A ceiling well "
                       "below 1 means part of the low ARI is arithmetic, not empirical.",
            "permutation_null": f"Category labels shuffled ({n_perm} draws) with fitted cluster "
                                "labels held fixed, so the null preserves both the true category "
                                "sizes and the observed cluster sizes.",
            "reading": "A run whose ARI does not exceed this null is not evidence of correspondence, "
                       "whatever its raw value. Conversely a low raw ARI paired with a low ceiling "
                       "is weaker evidence of ABSENCE than the raw number suggests.",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "true_category_sizes": list(true_sizes),
        "runs": rows,
    }
    op = os.path.join(out_dir, "clustering_scope_check.json")
    json.dump(payload, open(op, "w"), indent=2)
    print(f"\n[INFO] wrote {op}")


if __name__ == "__main__":
    fire.Fire(main)
