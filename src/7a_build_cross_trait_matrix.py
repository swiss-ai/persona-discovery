"""Assemble the 30x30 cross-trait steering matrix from the judge results.

Cell [i][j] is the steering preference rate when trait i's steered completions are judged for trait
j's expression, at one steering strength. Reads

  $SCRATCH/apertus-pulse/data/steering_eval/<checkpoint>/l1_<l1>/trait_<i>/eval_trait_<j>[_neg]/summary.json
      (src/6b_evaluate_steering.py; `_neg` = the same answer pairs judged for trait j's negative pole)
  data/plots/probe_geometry/<checkpoint>/geometry_l1_<l1>.json
      (src/4a_analyze_trait_geometry.py; the probe-direction cosine matrix)

and writes data/plots/probe_geometry/<checkpoint>/cross_trait_matrix.json with the matrices rate,
rate_neg, d = rate - rate_neg, n_pairs, n_unparseable and cosine_similarity.

    python -m src.7a_build_cross_trait_matrix --checkpoint apertus_v1.5_8B
"""
import os, json, datetime
import fire

from src.utils.paths import repo_path, data_path


def _cell(root, steered, judged, sign, strength):
    sub = f"eval_trait_{judged}" + ("_neg" if sign == "neg" else "")
    path = os.path.join(root, f"trait_{steered}", sub, "summary.json")
    return json.load(open(path))[strength]


def main(checkpoint: str = "apertus_v1.5_8B", l1: str = "0.0", strength: str = "0.02",
         n_traits: int = 30, eval_dir: str = None, out_dir: str = None):
    l1, strength = str(l1), str(strength)
    root = os.path.join(eval_dir or data_path("steering_eval"), checkpoint, f"l1_{l1}")
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    ids = list(range(1, n_traits + 1))

    rate, rate_neg, d, n_pairs, n_unparseable = [], [], [], [], []
    for i in ids:
        pos = [_cell(root, i, j, "pos", strength) for j in ids]
        neg = [_cell(root, i, j, "neg", strength) for j in ids]
        rate.append([c["steered_preferred_rate"] for c in pos])
        rate_neg.append([c["steered_preferred_rate"] for c in neg])
        d.append([p["steered_preferred_rate"] - q["steered_preferred_rate"] for p, q in zip(pos, neg)])
        n_pairs.append([c["n_pairs"] for c in pos])
        n_unparseable.append([c["n_unparseable"] for c in pos])
    pair_counts = {n for row in n_pairs for n in row}
    if len(pair_counts) != 1:
        raise ValueError(f"cells were judged on different numbers of answer pairs: {sorted(pair_counts)}")

    geometry = json.load(open(os.path.join(out_dir, f"geometry_l1_{l1}.json")))
    if geometry["trait_ids"] != ids:
        raise ValueError("trait order of the geometry file does not match")
    traits = json.load(open(repo_path("data", "traits", "all_traits.json")))["therapist_traits"]
    names = {t["trait_id"]: t["trait_name_pos"] for t in traits}
    categories = json.load(open(repo_path("data", "traits", "trait_categories.json")))["trait_categories"]

    payload = {
        "_meta": {
            "description": "Cross-trait steering matrix. Cell [i][j] = steered_preferred_rate when trait i's "
                           "steered completions are judged for trait j's expression. Asymmetric: [i][j] and "
                           "[j][i] judge different completions; this is not a similarity matrix.",
            "checkpoint": checkpoint,
            "l1": float(l1),
            "strength": float(strength),
            "num_pairs": pair_counts.pop(),
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "source": f"steering_eval/{checkpoint}/l1_{l1}/trait_<i>/eval_trait_<j>[_neg]/summary.json, "
                      f"strength {strength}",
            "geometry_counterpart": f"data/plots/probe_geometry/{checkpoint}/geometry_l1_{l1}.json -> "
                                    "mean_similarity_matrix (symmetric), copied as cosine_similarity",
            "row_index": "trait that was steered",
            "col_index": "trait the judge scored for",
            "rate_neg": "steered_preferred_rate when judged against the eval trait's negative pole (same "
                        "answer pairs, same A/B order; only the trait named in the prompt differs)",
            "d": "d = rate - rate_neg. Under pure judge quality bias rate == rate_neg, so d -> 0; genuine "
                 "trait expression drives the poles apart. Range [-1, 1].",
        },
        "trait_ids": ids,
        "trait_names": [names[i] for i in ids],
        "rate": rate,
        "n_pairs": n_pairs,
        "n_unparseable": n_unparseable,
        "cosine_similarity": geometry["mean_similarity_matrix"],
        "categories": [categories[str(i)] for i in ids],
        "rate_neg": rate_neg,
        "d": d,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cross_trait_matrix.json")
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"[INFO] wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
