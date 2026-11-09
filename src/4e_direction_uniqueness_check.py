"""How similar are probe directions for the same trait trained with different L1 penalties?

For every trait and layer, the script compares the probe weight vectors of the five L1 penalties
(cosine similarity) and reports their distribution over traits and layers, with two reference scales:
different traits at the same penalty, and random vectors of the same dimension. It compares geometry
only; whether such directions steer alike is measured in src/7e_steering_l1_comparison.py.

Everything compared is matched except the L1 penalty: same checkpoint, trait, layer, submodule,
pooling, set, and the same fixed-seed 70/30 split (`load_split` seeds on 42 regardless of L1).

Probes are trained with positive inputs labelled 1, so every weight vector points toward the positive
pole and a near-zero cosine is not an unresolved sign flip. The script reports the signed cosine and
the fraction of negative values as a check.

Reads the probe weights of all five L1 values ($SCRATCH/apertus-pulse/data/probes/<checkpoint>/...)
and writes data/plots/probe_geometry/<checkpoint>/direction_uniqueness.json.

    python -m src.4e_direction_uniqueness_check --checkpoint apertus_v1.5_8B
"""
import os, json, datetime, itertools
import numpy as np
import torch
import fire

from src.utils.paths import repo_path, data_path

L1S = ["0.0", "0.01", "0.1", "0.5", "1.0"]


def _w(path):
    """Weight vector of a LinearProbe state dict, as a flat numpy array."""
    return torch.load(path, map_location="cpu", weights_only=True)["w"].detach().cpu().numpy().reshape(-1)


def main(checkpoint: str = "apertus_v1.5_8B", set_name: str = "cal",
         submodule: str = "residual.post_mlp", pooling: str = "mean_pooling",
         n_traits: int = 30, out_dir: str = None):
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    root = data_path("probes", checkpoint)

    vecs, layers = {}, None
    for tid in range(1, n_traits + 1):
        for l1 in L1S:
            base = os.path.join(root, f"trait_{tid}", set_name, "log_probes", submodule, pooling, f"l1_{l1}")
            if not os.path.isdir(base):
                continue
            ls = sorted(int(f.split("_")[1].split(".")[0]) for f in os.listdir(base) if f.startswith("layer_"))
            layers = ls if layers is None else layers
            for L in ls:
                vecs[(tid, l1, L)] = _w(os.path.join(base, f"layer_{L}.pt"))
    print(f"[INFO] loaded {len(vecs)} probe vectors | {n_traits} traits x {len(L1S)} L1 x {len(layers)} layers")

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na and nb else np.nan

    # 1. Same trait, same layer, different L1 -- the quantity of interest.
    same, pairs = {}, list(itertools.combinations(L1S, 2))
    for a, b in pairs:
        v = [cos(vecs[(t, a, L)], vecs[(t, b, L)])
             for t in range(1, n_traits + 1) for L in layers
             if (t, a, L) in vecs and (t, b, L) in vecs]
        same[f"{a}_vs_{b}"] = v

    # 2. Reference scales: different traits at the same L1, and random vectors of the same dimension.
    diff = {}
    for l1 in L1S:
        v = [cos(vecs[(t1, l1, L)], vecs[(t2, l1, L)])
             for L in layers for t1, t2 in itertools.combinations(range(1, n_traits + 1), 2)
             if (t1, l1, L) in vecs and (t2, l1, L) in vecs]
        diff[l1] = v
    d_model = len(next(iter(vecs.values())))
    rng = np.random.default_rng(0)
    rand = [cos(rng.normal(size=d_model), rng.normal(size=d_model)) for _ in range(2000)]

    print(f"\nSAME trait, same layer, DIFFERENT L1 (n={len(next(iter(same.values())))} trait-layer pairs each)")
    print(f"{'comparison':<16}{'mean':>8}{'median':>8}{'sd':>7}{'p5':>7}{'p95':>7}{'frac<0':>8}")
    for k, v in same.items():
        v = np.array(v)
        print(f"{k:<16}{v.mean():>8.3f}{np.median(v):>8.3f}{v.std():>7.3f}"
              f"{np.percentile(v,5):>7.3f}{np.percentile(v,95):>7.3f}{(v<0).mean():>8.3f}")

    print("\nReference: DIFFERENT traits at the same L1")
    for k, v in diff.items():
        v = np.array(v)
        print(f"  l1={k:<5} mean={v.mean():+.3f}  median={np.median(v):+.3f}  sd={v.std():.3f}")
    ra = np.array(rand)
    print(f"  random {d_model}-d vectors: mean={ra.mean():+.4f}  sd={ra.std():.4f}  "
          f"|cos| 95th pct={np.percentile(np.abs(ra),95):.4f}")

    # 3. l1=0.0 vs l1=1.0 per layer: is near-orthogonality uniform with depth?
    print(f"\nSame trait, l1=0.0 vs l1=1.0, BY LAYER (mean over {n_traits} traits)")
    by_layer = {}
    for L in layers:
        v = [cos(vecs[(t, "0.0", L)], vecs[(t, "1.0", L)]) for t in range(1, n_traits + 1)
             if (t, "0.0", L) in vecs and (t, "1.0", L) in vecs]
        by_layer[L] = float(np.mean(v))
    sel = [l for l in layers if l % 4 == 0 or l == layers[-1]]
    print("  " + "  ".join(f"L{L}={by_layer[L]:+.2f}" for L in sel))

    # 4. Per trait, to show whether a few traits drive the result.
    per_trait = {}
    for t in range(1, n_traits + 1):
        v = [cos(vecs[(t, "0.0", L)], vecs[(t, "1.0", L)]) for L in layers if (t, "0.0", L) in vecs]
        per_trait[t] = float(np.mean(v))
    vals = np.array(list(per_trait.values()))
    print(f"\nPer-trait mean cosine(l1=0.0, l1=1.0): min={vals.min():+.3f} median={np.median(vals):+.3f} "
          f"max={vals.max():+.3f}; traits with mean cosine < 0.10: {(vals<0.10).sum()}/{n_traits}")

    # 5. Norm and near-zero fraction per L1 (scale and sparsity of the vectors).
    print("\nNorm and near-zero fraction at layer 16 (trait-averaged)")
    for l1 in L1S:
        ws = [vecs[(t, l1, 16)] for t in range(1, n_traits + 1) if (t, l1, 16) in vecs]
        if not ws:
            continue
        nz = np.mean([(np.abs(w) < 1e-3 * np.abs(w).max()).mean() for w in ws])
        print(f"  l1={l1:<5} mean ||w||={np.mean([np.linalg.norm(w) for w in ws]):>8.3f}   near-zero frac={nz:.3f}")

    payload = {
        "_meta": {
            "description": "Distribution of cosine similarity between probe directions for the SAME "
                           "trait and layer trained at different L1 penalties.",
            "matched": "checkpoint, trait, layer, submodule, pooling, set, and the fixed-seed 70/30 "
                       "split are identical; only the L1 penalty differs.",
            "sign": "Positive inputs are labelled 1, so every weight vector is oriented toward the "
                    "positive pole by construction. `frac_negative` is the check that a near-zero "
                    "cosine is genuine near-orthogonality and not an unresolved sign flip.",
            "reading": "Near-zero cosine between two directions that both classify well and steer "
                       "comparably supports 'many linear directions separate these conditions', NOT "
                       "'the model has no trait representation'. Steering equivalence is measured "
                       "separately (src/7e_steering_l1_comparison.py).",
            "caveat": "Cosine in the raw Euclidean space. Two directions can be near-orthogonal and "
                      "still act similarly on the model if the intervening geometry is anisotropic; "
                      "this is why the behavioural comparison is the load-bearing evidence.",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "checkpoint": checkpoint, "n_traits": n_traits, "layers": layers, "d_model": d_model,
        "same_trait_different_l1": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                        "sd": float(np.std(v)), "frac_negative": float(np.mean(np.array(v) < 0)),
                                        "n": len(v)} for k, v in same.items()},
        "different_traits_same_l1": {k: {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                         "sd": float(np.std(v))} for k, v in diff.items()},
        "random_baseline": {"mean": float(ra.mean()), "sd": float(ra.std()),
                            "abs_95th_pct": float(np.percentile(np.abs(ra), 95))},
        "l1_0_vs_1_by_layer": by_layer,
        "l1_0_vs_1_per_trait": per_trait,
    }
    os.makedirs(out_dir, exist_ok=True)
    op = os.path.join(out_dir, "direction_uniqueness.json")
    json.dump(payload, open(op, "w"), indent=2)
    print(f"\n[INFO] wrote {op}")


if __name__ == "__main__":
    fire.Fire(main)
