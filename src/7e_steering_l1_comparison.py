"""Does the L1 penalty of the probe change how well its direction steers?

Compares steering the same trait with the probe directions of `l1=0.0`, `l1=0.1` and `l1=1.0` at
matched strengths, via the steering preference rate (SPR) of the judge (src/6b_evaluate_steering.py).
src/4e_direction_uniqueness_check.py compares the same directions geometrically.

Design:
  observation  one trait (`n_traits`, 30 in the thesis), giving a paired SPR triple per strength
  pairing      same trait, questions, judge and strength; only the L1 penalty of the steered probe
               differs
  intervention the whole set of per-layer directions is exchanged at once (all layers with a probe;
               32 for apertus_v1.5_8B), so this is not a per-layer comparison
  reported     mean SPR per L1 over the traits with results, the paired mean/median difference, and
               the number of traits for which each direction set gives the higher SPR

The comparison is descriptive. Two-sided Wilcoxon signed-rank tests (with a Bonferroni family over
strengths 0.005 and 0.01) are also stored in the output (`wilcoxon_*`, `bonferroni_*`).

At +-0.02, many completions of the penalised directions are degenerate (in the thesis data, only 9-13
of 30 traits stayed coherent), so the SPR there mixes trait expression with degeneration. These
strengths are flagged as not comparable across L1 (`comparable_across_l1`); the means are still taken
over all traits, and the fraction of coherent traits is reported (`coherent_trait_fraction`).

Reads steering_eval/ and steering_coherence/ under $SCRATCH/apertus-pulse/data (l1 0.0, 0.1, 1.0) and
writes data/plots/probe_geometry/<checkpoint>/steering_l1_comparison.json.

    python -m src.7e_steering_l1_comparison
"""
import os, json, datetime
import numpy as np
import fire
from scipy.stats import wilcoxon

from src.utils.paths import repo_path, data_path

L1S = ["0.0", "0.1", "1.0"]
SHARED_GRID = ["-0.02", "-0.01", "-0.005", "0.001", "0.002", "0.005", "0.01", "0.02"]
FAMILY = ["0.005", "0.01"]          # Bonferroni family of the paired tests
NOT_COMPARABLE = {"-0.02", "0.02"}  # many degenerate completions at these strengths (see module docstring)


def _spr(root, ckpt, l1, trait, strength):
    p = os.path.join(root, ckpt, f"l1_{l1}", f"trait_{trait}", f"eval_trait_{trait}", "summary.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    v = d.get(strength)
    return float(v["steered_preferred_rate"]) if v else None


def _incoh(root_c, ckpt, l1, trait, strength):
    p = os.path.join(root_c, ckpt, f"l1_{l1}", f"trait_{trait}", "summary.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return float(d[strength]["incoherent_rate_absolute"]) if strength in d else None


def main(checkpoint: str = "apertus_v1.5_8B", n_traits: int = 30,
         eval_dir: str = None, coherence_dir: str = None, out_dir: str = None,
         coherence_cut: float = 0.05):
    eval_dir = eval_dir or data_path("steering_eval")
    coherence_dir = coherence_dir or data_path("steering_coherence")
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    traits = list(range(1, n_traits + 1))

    spr = {l1: {s: [_spr(eval_dir, checkpoint, l1, t, s) for t in traits] for s in SHARED_GRID} for l1 in L1S}
    coh = {l1: {s: [_incoh(coherence_dir, checkpoint, l1, t, s) for t in traits] for s in SHARED_GRID} for l1 in L1S}

    print(f"[INFO] {checkpoint}: {n_traits} traits x {len(L1S)} L1 x {len(SHARED_GRID)} strengths")
    print(f"\n{'strength':>9}" + "".join(f"{'l1='+l:>10}" for l in L1S) + "   coherent-trait fraction (<%.0f%% incoh)" % (coherence_cut * 100))
    coherent_frac = {}
    for s in SHARED_GRID:
        row = f"{s:>9}"
        for l1 in L1S:
            v = [x for x in spr[l1][s] if x is not None]
            row += f"{np.mean(v):>10.3f}"
        fr = []
        for l1 in L1S:
            c = [x for x in coh[l1][s] if x is not None]
            fr.append(sum(1 for x in c if x < coherence_cut) / max(len(c), 1))
        coherent_frac[s] = dict(zip(L1S, fr))
        flag = "  <- NOT comparable across L1" if s in NOT_COMPARABLE else ""
        print(row + "   " + " ".join(f"{f:.2f}" for f in fr) + flag)

    tests = []
    print(f"\n{'strength':>9}{'vs':>7}{'mean diff':>11}{'favouring l1=0':>16}{'p (Wilcoxon)':>14}  Bonferroni 0.0125")
    for s in SHARED_GRID:
        for l1 in ["0.1", "1.0"]:
            a, b = spr["0.0"][s], spr[l1][s]
            pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
            diff = [x - y for x, y in pairs]
            res = wilcoxon(diff)
            in_family = s in FAMILY
            rec = {"strength": s, "vs_l1": l1, "n_pairs": len(pairs),
                   "mean_difference": float(np.mean(diff)),
                   "median_difference": float(np.median(diff)),
                   "n_favouring_l1_0": int(sum(1 for d in diff if d > 0)),
                   "wilcoxon_statistic": float(res.statistic), "wilcoxon_p_two_sided": float(res.pvalue),
                   "in_bonferroni_family": in_family,
                   "survives_bonferroni_0.0125": bool(in_family and res.pvalue < 0.05 / 4),
                   "comparable_across_l1": s not in NOT_COMPARABLE}
            tests.append(rec)
            if in_family:
                mark = "SURVIVES" if rec["survives_bonferroni_0.0125"] else "does not survive"
                print(f"{s:>9}{l1:>7}{np.mean(diff):>+11.3f}{rec['n_favouring_l1_0']:>13}/30{res.pvalue:>14.4f}  {mark}")

    payload = {
        "_meta": {
            "description": "Behavioural half of the direction non-uniqueness finding: do near-orthogonal "
                           "probe solutions for the same trait steer it comparably?",
            "reporting": "Descriptive: mean_spr, mean_difference, median_difference and n_favouring_l1_0. "
                         "The wilcoxon_* and bonferroni_* fields are computed for completeness; no "
                         "inferential claim rests on them.",
            "observation_unit": "one trait (n=30); paired across L1 at matched strength",
            "pairing": "same trait, same 50 held-out questions, same judge, same strength; only the L1 "
                       "penalty of the steered probe differs",
            "intervention_granularity": "the FULL set of per-layer directions is replaced, all 32 layers "
                                        "at once. NOT a per-layer comparison; per-layer causal "
                                        "equivalence was never tested.",
            "test": "two-sided Wilcoxon signed-rank on paired per-trait differences (scipy.stats.wilcoxon)",
            "not_an_equivalence_test": "Failing to reject is absence of a detected difference, not "
                                       "demonstrated equivalence. No TOST or CI on the paired difference "
                                       "was computed.",
            "multiplicity": "The four tests at strengths 0.005 and 0.01 against l1 in {0.1, 1.0} form one "
                            "family, with Bonferroni threshold 0.05/4 = 0.0125 (survives_bonferroni_0.0125). "
                            "Tests at other strengths are not part of the family.",
            "not_comparable": "The +-0.02 columns are not comparable across L1: their means include all "
                              "available trait rates without coherence filtering, and many completions of "
                              "the regularised directions are degenerate there, which complicates reading "
                              "the rates as trait expression. Flagged per row as comparable_across_l1=false.",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "checkpoint": checkpoint, "n_traits": n_traits, "l1_values": L1S,
        "strengths": SHARED_GRID, "bonferroni_family": FAMILY, "bonferroni_threshold": 0.05 / 4,
        "coherence_cut": coherence_cut,
        "mean_spr": {l1: {s: float(np.mean([x for x in spr[l1][s] if x is not None])) for s in SHARED_GRID} for l1 in L1S},
        "coherent_trait_fraction": coherent_frac,
        "tests": tests,
    }
    os.makedirs(out_dir, exist_ok=True)
    op = os.path.join(out_dir, "steering_l1_comparison.json")
    json.dump(payload, open(op, "w"), indent=2)
    print(f"\n[INFO] wrote {op}")


if __name__ == "__main__":
    fire.Fire(main)
