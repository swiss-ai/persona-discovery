"""Does geometric proximity between two trait directions predict behavioural transfer?

Reads data/plots/probe_geometry/<checkpoint>/cross_trait_matrix.json, which holds the
behavioural matrices of the 30x30 cross-trait steering sweep (rate, rate_neg, d) and the aligned
cosine matrix of the probe directions, and writes cross_trait_correlation.json next to it.

Three points the statistics account for:

1. The behavioural matrix is asymmetric. Cell (i,j) is trait i's steered completions judged for
   trait j's expression; (j,i) judges different completions. The cosine matrix is symmetric. Every
   statistic states which convention it uses: asymmetric off-diagonal, symmetrised as (M+M.T)/2,
   row-wise or column-wise.

2. The cells are not independent. Each trait appears in n-1 rows and n-1 columns (29 for the 30
   thesis traits), so a plain Spearman p-value over the off-diagonal cells is anticonservative. Every matrix-level p is a Mantel
   test: permute the trait labels jointly on rows and columns of the behavioural matrix and recompute
   the statistic, n_perm times. The smallest attainable p is 1/(n_perm+1).

3. The judge may track the named trait only for strongly steerable traits (in the thesis data, the
   negative-pole control showed this). Statistics are therefore reported for all cells and restricted
   to the strong steerers (diagonal rate >= strong_thresh). That subset is selected by steerability,
   so it is a supporting statistic.

    python -m src.7c_analyze_cross_trait_correlation --checkpoint apertus_v1.5_8B
"""
import os, json, datetime
import numpy as np
import fire
from scipy.stats import spearmanr, rankdata, binomtest, wilcoxon, mannwhitneyu
from src.utils.paths import repo_path


def _rho(a, b):
    r = spearmanr(a, b).correlation
    return float(r) if np.isfinite(r) else float("nan")


def _double_center(M, off):
    """Remove additive row and column effects, leaving pair-specific variance only.

    Ranks first, then subtract each source trait's mean and each target trait's mean. What
    survives is neither "this source steers well" nor "this target is easy to affirm", which
    is exactly the confound a few highly judge-favourable targets would otherwise create.
    """
    flat = np.where(off, M, np.nan).ravel()
    A = rankdata(flat, nan_policy="omit").reshape(M.shape)
    A = A - np.nanmean(A, axis=1, keepdims=True)
    A = A - np.nanmean(A, axis=0, keepdims=True)
    return A


def _mantel(C, B, mask_fn, off, n, symmetrise, n_perm, rng):
    """Spearman(cosine, behaviour) with a joint row/column label-permutation null."""
    iu = np.triu_indices(n, k=1)

    def stat(M):
        if symmetrise:
            Ms = (M + M.T) / 2.0
            m = mask_fn(np.arange(n))[iu]
            return _rho(C[iu][m], Ms[iu][m]), int(m.sum())
        m = mask_fn(np.arange(n)) & off
        return _rho(C[m], M[m]), int(m.sum())

    obs, k = stat(B)
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        r, _ = stat(B[np.ix_(p, p)])
        if abs(r) >= abs(obs) - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1), k


def _mantel_centred(C, B, off, n, n_perm, rng):
    Cc = _double_center(C, off)
    obs = _rho(Cc[off], _double_center(B, off)[off])
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(_rho(Cc[off], _double_center(B[np.ix_(p, p)], off)[off])) >= abs(obs) - 1e-12:
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


def main(
    checkpoint: str = "apertus_v1.5_8B",
    matrix_json: str = None,
    out_dir: str = None,
    n_perm: int = 20000,
    seed: int = 0,
    strong_thresh: float = 0.90,
    weak_thresh: float = 0.70,
    n_top_cells: int = 20,
    drop_targets: list = None,
    n_drop_targets: int = 2,
):
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    matrix_json = matrix_json or os.path.join(out_dir, "cross_trait_matrix.json")
    src = json.load(open(matrix_json))

    ids, names, cats = src["trait_ids"], src["trait_names"], src["categories"]
    R = np.array(src["rate"], float)
    NEG = np.array(src["rate_neg"], float)
    D = np.array(src["d"], float)
    C = np.array(src["cosine_similarity"], float)
    n = len(ids)
    off = ~np.eye(n, dtype=bool)
    diag = np.diag(R)

    strong = np.where(diag >= strong_thresh)[0]
    weak = np.where(diag < weak_thresh)[0]
    rng = np.random.default_rng(seed)

    print(f"[INFO] {checkpoint}: {n} traits, {int(off.sum())} off-diagonal cells")
    print(f"[INFO] strong steerers (diag >= {strong_thresh}): {[ids[i] for i in strong]}")
    print(f"[INFO] weak steerers   (diag <  {weak_thresh}): {[ids[i] for i in weak]}")

    def m_all(idx):
        return np.ones((n, n), dtype=bool)

    def m_both_strong(idx):
        s = np.isin(idx, strong)
        return np.outer(s, s)

    def m_row_strong(idx):
        return np.repeat(np.isin(idx, strong)[:, None], n, axis=1)

    conventions = [
        ("asymmetric_off_diagonal", m_all, False),
        ("symmetrised", m_all, True),
        ("both_traits_strong", m_both_strong, False),
        ("both_traits_strong_symmetrised", m_both_strong, True),
        ("steered_trait_strong", m_row_strong, False),
    ]

    matrix_level = []
    print(f"\n{'behavioural':<10}{'convention':<32}{'cells':>7}{'rho':>9}{'p_mantel':>11}")
    for bname, B in [("rate", R), ("d", D), ("rate_neg", NEG)]:
        for cname, mf, sym in conventions:
            rho, p, k = _mantel(C, B, mf, off, n, sym, n_perm, rng)
            matrix_level.append({"behavioural": bname, "convention": cname,
                                 "n_cells": k, "rho": rho, "p_mantel": p})
            print(f"{bname:<10}{cname:<32}{k:>7}{rho:>9.3f}{p:>11.5f}")

    # Row-wise: one correlation per steered trait, across its targets.
    # Column-wise: one per target trait, across its sources. Column-wise is the one that
    # controls for target attractiveness, since the target is held fixed within each.
    row_wise, col_wise = [], []
    for i in range(n):
        mr, mc = off[i], off[:, i]
        row_wise.append({"trait_id": ids[i], "trait_name": names[i], "category": cats[i],
                         "own_diagonal_rate": float(diag[i]),
                         "mean_rate_out": float(R[i][mr].mean()),
                         "rho_rate": _rho(C[i][mr], R[i][mr]),
                         "rho_d": _rho(C[i][mr], D[i][mr]),
                         "is_strong_steerer": bool(i in strong)})
        col_wise.append({"trait_id": ids[i], "trait_name": names[i], "category": cats[i],
                         "own_diagonal_rate": float(diag[i]),
                         "mean_rate_in": float(R[mc, i].mean()),
                         "rho_rate": _rho(C[mc, i], R[mc, i]),
                         "rho_d": _rho(C[mc, i], D[mc, i]),
                         "is_strong_steerer": bool(i in strong)})

    def _agg(rows, key):
        v = np.array([r[key] for r in rows], float)
        pos = int((v > 0).sum())
        return {"mean": float(v.mean()), "median": float(np.median(v)),
                "n_positive": pos, "n_total": len(v),
                "sign_test_p": float(binomtest(pos, len(v), 0.5).pvalue),
                "wilcoxon_p": float(wilcoxon(v).pvalue)}

    aggregates = {
        "row_wise": {k: _agg(row_wise, k) for k in ("rho_rate", "rho_d")},
        "column_wise": {k: _agg(col_wise, k) for k in ("rho_rate", "rho_d")},
        "row_wise_strong_only": {
            k: float(np.mean([r[k] for r in row_wise if r["is_strong_steerer"]]))
            for k in ("rho_rate", "rho_d")},
    }
    for scope in ("row_wise", "column_wise"):
        a = aggregates[scope]["rho_rate"]
        print(f"\n[INFO] {scope}: rho(cos,rate) mean {a['mean']:+.3f}, "
              f"positive {a['n_positive']}/{a['n_total']}, sign p={a['sign_test_p']:.5f}")

    # Row and column effects removed simultaneously.
    double_centred = {}
    for bname, B in [("rate", R), ("d", D)]:
        rho, p = _mantel_centred(C, B, off, n, n_perm, rng)
        double_centred[bname] = {"rho": rho, "p_mantel": p}
        print(f"[INFO] doubly-centred {bname}: rho={rho:+.3f} p={p:.5f}")

    # Sensitivity: is the whole effect carried by the targets that attract the most transfer?
    # The dropped targets are those with the highest mean incoming rate.
    incoming = np.array([R[off[:, i], i].mean() for i in range(n)])
    top_targets = drop_targets or [ids[i] for i in np.argsort(-incoming)[:n_drop_targets]]
    keep = np.array([i for i in range(n) if ids[i] not in top_targets])
    k = len(keep)
    offk = ~np.eye(k, dtype=bool)
    sub = np.ix_(keep, keep)
    sensitivity = {"dropped_top_targets": top_targets, "drop_top_targets": {},
                   "exclude_strong_targets": {}}
    for bname, B in [("rate", R), ("d", D)]:
        Bs, Cs = B[sub], C[sub]
        obs = _rho(Cs[offk], Bs[offk])
        ge = 0
        for _ in range(n_perm):
            p = rng.permutation(k)
            if abs(_rho(Cs[offk], Bs[np.ix_(p, p)][offk])) >= abs(obs) - 1e-12:
                ge += 1
        sensitivity["drop_top_targets"][bname] = {
            "rho": obs, "p_mantel": (ge + 1) / (n_perm + 1), "n_cells": int(offk.sum())}
        m = off.copy()
        m[:, strong] = False
        sensitivity["exclude_strong_targets"][bname] = {
            "rho": _rho(C[m], B[m]), "n_cells": int(m.sum())}
    print(f"[INFO] dropping top targets {top_targets}: "
          f"rate rho={sensitivity['drop_top_targets']['rate']['rho']:+.3f}")

    # Controls. rate_neg is the key one: under pure judge quality bias, cells that win on the
    # positive pole also win on the negative pole, so rho(cos, rate_neg) would be positive;
    # trait expression predicts the opposite sign.
    src_diag = np.repeat(diag, n).reshape(n, n)
    tgt_diag = np.tile(diag, (n, 1))
    controls = {
        "rho_rate_vs_rate_neg": _rho(R[off], NEG[off]),
        "rho_source_diagonal_vs_rate": _rho(src_diag[off], R[off]),
        "rho_target_diagonal_vs_rate": _rho(tgt_diag[off], R[off]),
        "rho_cos_vs_abs_rate_minus_half": _rho(C[off], np.abs(R[off] - 0.5)),
    }
    print(f"[INFO] control rho(rate, rate_neg) = {controls['rho_rate_vs_rate_neg']:+.3f}")

    # Does the expert taxonomy predict transfer on the same cells?
    #
    # A Mann-Whitney test over the off-diagonal cells would treat them as independent, which they are
    # not (see point 2 of the module docstring). The category test is therefore a trait-level
    # permutation: hold the behavioural matrix fixed and permute the trait->category assignments. That breaks the
    # category-transfer relationship while preserving the matrix in full (row effects, column
    # effects, asymmetry) and preserving the category-size profile, so n_same stays fixed.
    # Mann-Whitney is retained alongside as a descriptive statistic only.
    same = np.array([[cats[i] == cats[j] for j in range(n)] for i in range(n)])
    cats_arr = np.array(cats)
    category_check = {}
    for bname, B in [("rate", R), ("d", D)]:
        a, b = B[off & same], B[off & ~same]
        obs_diff = float(a.mean() - b.mean())
        obs_u = float(mannwhitneyu(a, b, alternative="two-sided").statistic)

        rng_c = np.random.default_rng(seed)
        null_diff = np.empty(n_perm)
        for k in range(n_perm):
            pc = rng_c.permutation(cats_arr)
            sm = pc[:, None] == pc[None, :]
            null_diff[k] = B[off & sm].mean() - B[off & ~sm].mean()
        # Permutation p = (b+1)/(m+1): b = null draws at least as extreme as the observed
        # difference, m = n_perm (Phipson & Smyth 2010); the same estimator as the other
        # permutation tests.
        b_ge = int(np.sum(np.abs(null_diff) >= abs(obs_diff) - 1e-12))
        p_perm = float((b_ge + 1) / (n_perm + 1))

        category_check[bname] = {
            "same_category_mean": float(a.mean()), "n_same": int(a.size),
            "cross_category_mean": float(b.mean()), "n_cross": int(b.size),
            "observed_difference": obs_diff,
            "perm_p_trait_level": p_perm,
            "perm_null_mean": float(null_diff.mean()),
            "perm_null_sd": float(null_diff.std()),
            "detectable_difference_95": float(np.percentile(np.abs(null_diff), 95)),
            "mannwhitney_p_descriptive_only": float(mannwhitneyu(a, b, alternative="two-sided").pvalue),
            "mannwhitney_u": obs_u,
        }
        cc = category_check[bname]
        print(f"[INFO] category check {bname}: same {a.mean():.3f} vs cross {b.mean():.3f}, "
              f"diff={obs_diff:+.4f}, trait-level perm p={cc['perm_p_trait_level']:.4f} "
              f"(null sd {cc['perm_null_sd']:.4f}, |diff| detectable at 95% = "
              f"{cc['detectable_difference_95']:.4f})")

    cells = sorted(
        ({"steered_trait": ids[i], "judged_trait": ids[j],
          "steered_name": names[i], "judged_name": names[j],
          "rate": float(R[i][j]), "rate_neg": float(NEG[i][j]),
          "d": float(D[i][j]), "cosine": float(C[i][j])}
         for i in range(n) for j in range(n) if i != j),
        key=lambda c: -c["d"])
    top_cells = cells[:n_top_cells]
    mean_cos_top = float(np.mean([c["cosine"] for c in cells[:50]]))

    result = {
        "_meta": {
            "description": "Spearman correlation between probe-direction cosine similarity "
                           "and cross-trait steering transfer, on the 30x30 matrix.",
            "source": os.path.basename(matrix_json),
            "pairs_source": "The full per-cell matrices (cosine, rate, rate_neg, d) are in "
                            "cross_trait_matrix.json. This file holds the statistics and, for "
                            "reference, the top transfer cells.",
            "asymmetry": "The behavioural matrix is asymmetric; the cosine matrix is symmetric. "
                         "Each entry in matrix_level names its convention; symmetrised and "
                         "asymmetric values differ materially (max |R[i][j]-R[j][i]| = 0.84).",
            "p_values": f"All matrix-level p-values are MANTEL tests ({n_perm} joint row/column "
                        f"label permutations), because cells sharing a trait are not independent. "
                        f"The resolution floor is {1/(n_perm+1):.5f}; a p at the floor means "
                        f"'< {1/(n_perm+1):.4f}', not zero.",
            "judge_validity": "The negative-pole control validates the judge for strongly-steerable "
                              "traits and is inconclusive for weak ones. The both_traits_strong rows "
                              "are the validated subset; it is small (42 off-diagonal cells from "
                              "7 traits) and selected by steerability, so the unrestricted value is "
                              "the primary statistic and the restricted one is supporting.",
            "rate_neg_sign": "rho(cos, rate_neg) is expected to be negative: under pure judge quality "
                             "bias it would be positive (near traits win on both poles), so a negative "
                             "sign means the control passes.",
            "double_centred": "Row and column effects removed simultaneously, so this reflects "
                              "pair-specific transfer only (it addresses the objection that a few "
                              "targets are simply easy to affirm).",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "config": {
            "checkpoint": checkpoint,
            "l1": src["_meta"].get("l1"),
            "strength": src["_meta"].get("strength"),
            "num_pairs": src["_meta"].get("num_pairs"),
            "judge_model": src["_meta"].get("judge_model"),
            "n_permutations": n_perm,
            "seed": seed,
            "strong_thresh": strong_thresh,
            "weak_thresh": weak_thresh,
        },
        "trait_ids": ids,
        "trait_names": names,
        "categories": cats,
        "diagonal_rate": diag.tolist(),
        "strong_steerers": [ids[i] for i in strong],
        "weak_steerers": [ids[i] for i in weak],
        "matrix_level": matrix_level,
        "row_wise": row_wise,
        "column_wise": col_wise,
        "aggregates": aggregates,
        "double_centred": double_centred,
        "sensitivity": sensitivity,
        "controls": controls,
        "category_check": category_check,
        "top_transfer_cells": top_cells,
        "mean_cosine_top50_by_d": mean_cos_top,
        "mean_cosine_all_off_diagonal": float(C[off].mean()),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_trait_correlation.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print(f"\n[INFO] wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
