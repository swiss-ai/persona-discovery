"""Score the coherence / degeneration of steered completions (src/5a_steer_trait.py, src/5b_steer_angular.py).

CPU-only and model-free: heuristics (length, character repetition) plus standard degeneration metrics
(distinct-n, max n-gram repeat, zlib ratio, self-repetition across the completions of a condition).

Two incoherence rates are reported per strength:

  absolute  fraction of completions failing fixed thresholds. Comparable across models, but if a
            model's unsteered output already fails them (in the thesis data, apertus_mini_5B_base),
            the rate is high at every strength and hides steering effects.
  relative  fraction below this trait's own baseline: distinct_4 under the `baseline_percentile`-th
            percentile of the unsteered completions. The baseline therefore flags about that fraction
            itself (exactly only without ties); `excess_rate` subtracts it.

Reads  <steering_dir>/<checkpoint>/l1_<l1>/trait_<id>/{baseline,strength_*}.json
Writes <out_dir>/<checkpoint>/l1_<l1>/trait_<id>/{<tag>.json, summary.json}, where <tag> is
`baseline` or the strength value, i.e. a separate folder from the steering files.
Two-trait output (pair_<A>_<B>/angle_*.json): set input_dir, output_dir and file_prefix=angle. If
output_dir is not set, the scores are written into input_dir, and the baseline scores then replace
the baseline completions in input_dir/baseline.json.

Usage:
    python -m src.6a_score_coherence trait_id=1 checkpoint=apertus_v1.5_8B
    python -m src.6a_score_coherence trait_id=all checkpoint=apertus_v1.5_8B
    python -m src.6a_score_coherence input_dir=<...>/pair_7_11 output_dir=<...>/pair_7_11 file_prefix=angle
"""
import os, json, re, glob
import hydra
from omegaconf import DictConfig

from src.utils.paths import ensure_scratch_env
from tqdm import tqdm

from src.utils.coherence_utils import (
    completion_flags,
    completion_metrics,
    percentile,
    self_repetition,
)


def _find_condition_files(trait_dir: str, prefix: str = "strength"):
    """[(value, tag, path)] for baseline + every condition file, sorted by value.

    `prefix` is the filename stem: "strength" for 5a_steer_trait output, "angle" for 5b_steer_angular output.
    """
    out = [(0.0, "baseline", os.path.join(trait_dir, "baseline.json"))]
    for path in glob.glob(os.path.join(trait_dir, f"{prefix}_*.json")):
        m = re.match(rf"^{re.escape(prefix)}_(-?\d*\.?\d+)\.json$", os.path.basename(path))
        if m:
            out.append((float(m.group(1)), m.group(1), path))
    return sorted([o for o in out if os.path.exists(o[2])], key=lambda x: x[0])


def _mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def score_trait(trait_dir: str, out_dir: str, cfg: DictConfig) -> dict:
    conditions = _find_condition_files(trait_dir, str(cfg.get("file_prefix", "strength")))
    if not conditions:
        return {}
    os.makedirs(out_dir, exist_ok=True)

    # Score every condition first; the baseline defines the relative threshold.
    scored, baseline_d4 = {}, []
    for strength, tag, path in conditions:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        rows = []
        for it in items:
            c = it.get("completion", "")
            m = completion_metrics(c)
            fl = completion_flags(
                c,
                min_words=cfg.min_words,
                max_char_rep=cfg.max_char_rep,
                min_distinct_4=cfg.min_distinct_4,
            )
            rows.append({"question": it.get("question"), **m, **fl})
        scored[tag] = (strength, rows, [it.get("completion", "") for it in items])
        if tag == "baseline":
            baseline_d4 = [r["distinct_4"] for r in rows]

    # Relative threshold: the baseline's own low-end percentile of distinct_4.
    rel_thr = percentile(baseline_d4, cfg.baseline_percentile) if baseline_d4 else float("nan")

    summary = {"_meta": {
        "relative_threshold_distinct_4": rel_thr,
        "baseline_percentile": cfg.baseline_percentile,
        "min_words": cfg.min_words,
        "max_char_rep": cfg.max_char_rep,
        "min_distinct_4": cfg.min_distinct_4,
    }}

    for tag, (strength, rows, comps) in scored.items():
        n = len(rows)
        for r in rows:
            r["below_baseline"] = bool(r["distinct_4"] < rel_thr) if rel_thr == rel_thr else False
        abs_rate = sum(r["incoherent"] for r in rows) / n if n else float("nan")
        rel_rate = sum(r["below_baseline"] for r in rows) / n if n else float("nan")

        with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        summary[tag] = {
            "strength": strength,
            "n": n,
            "incoherent_rate_absolute": abs_rate,
            "incoherent_rate_relative": rel_rate,
            "excess_rate": (rel_rate - cfg.baseline_percentile / 100.0) if rel_rate == rel_rate else float("nan"),
            "rate_empty_or_short": sum(r["empty_or_short"] for r in rows) / n if n else float("nan"),
            "rate_degenerate_repetition": sum(r["degenerate_repetition"] for r in rows) / n if n else float("nan"),
            "rate_low_distinct_4": sum(r["low_distinct_4"] for r in rows) / n if n else float("nan"),
            "mean_n_chars": _mean([r["n_chars"] for r in rows]),
            "mean_distinct_2": _mean([r["distinct_2"] for r in rows]),
            "mean_distinct_4": _mean([r["distinct_4"] for r in rows]),
            "mean_zlib_ratio": _mean([r["zlib_ratio"] for r in rows]),
            "mean_max_4gram_repeat": _mean([r["max_4gram_repeat"] for r in rows]),
            "self_repetition": self_repetition(comps),
        }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


# The Hydra configs interpolate ${oc.env:SCRATCH}, which OmegaConf resolves when the
# decorator below loads the config. Resolving it here first turns an unset SCRATCH into
# the clear error of utils/paths.py.
ensure_scratch_env()


@hydra.main(config_path="../configs", config_name="score_coherence", version_base=None)
def main(cfg: DictConfig) -> None:
    # input_dir/output_dir score one directory of a differently-shaped producer
    # (5b_steer_angular's pair_<A>_<B>/); file_prefix switches strength_*.json -> angle_*.json.
    if cfg.get("input_dir", None):
        s = score_trait(str(cfg.input_dir), str(cfg.output_dir or cfg.input_dir), cfg)
        for tag in sorted((k for k in s if k != "_meta"), key=lambda k: s[k]["strength"]):
            v = s[tag]
            print(f"  {tag:>12}: absolute={v['incoherent_rate_absolute']:.3f} "
                  f"relative={v['incoherent_rate_relative']:.3f} mean_distinct_4={v['mean_distinct_4']:.3f}")
        print(f"[INFO] wrote coherence scores -> {cfg.output_dir or cfg.input_dir}")
        return

    root = os.path.join(cfg.steering_dir, cfg.checkpoint, f"l1_{cfg.l1}")
    if str(cfg.trait_id).lower() == "all":
        traits = sorted(
            int(os.path.basename(p).split("_")[1])
            for p in glob.glob(os.path.join(root, "trait_*"))
            if os.path.isdir(p)
        )
    else:
        traits = [int(cfg.trait_id)]

    print(f"[INFO] scoring coherence | checkpoint={cfg.checkpoint} | traits={len(traits)}")
    done = 0
    for t in tqdm(traits, desc="traits"):
        trait_dir = os.path.join(root, f"trait_{t}")
        out_dir = os.path.join(cfg.out_dir, cfg.checkpoint, f"l1_{cfg.l1}", f"trait_{t}")
        s = score_trait(trait_dir, out_dir, cfg)
        if not s:
            print(f"[WARN] no completion files for trait {t} in {trait_dir}")
            continue
        done += 1
        if len(traits) == 1:
            for tag in sorted((k for k in s if k != "_meta"), key=lambda k: s[k]["strength"]):
                v = s[tag]
                print(f"  strength {v['strength']:>7}: absolute={v['incoherent_rate_absolute']:.3f} "
                      f"relative={v['incoherent_rate_relative']:.3f} excess={v['excess_rate']:+.3f} "
                      f"mean_distinct_4={v['mean_distinct_4']:.3f}")
    print(f"[INFO] wrote coherence scores for {done}/{len(traits)} traits -> {os.path.join(cfg.out_dir, cfg.checkpoint)}")


if __name__ == "__main__":
    main()
