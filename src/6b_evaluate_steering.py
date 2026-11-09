"""Judge steered answers against the unsteered baseline with an LLM, in pairwise comparisons.

For one steered trait (`trait_id`), every steered answer is shown to the judge together with the
baseline answer to the same question, in random order, and the judge picks the answer that better
expresses an evaluated trait. `eval_trait_id` (a trait id or a list; default: the steered trait)
selects the trait definitions to judge for, and `eval_sign=neg` judges for their negative poles
instead. Per strength, summary.json reports how often the steered answer was preferred. Output:
$SCRATCH/apertus-pulse/data/steering_eval/<checkpoint>/l1_<l1>/trait_<id>/eval_trait_<j>[_neg]/.
Two-trait steering from 5b_steer_angular is judged with input_dir, output_dir and file_prefix=angle.

Usage:
    python -m src.6b_evaluate_steering trait_id=24
    python -m src.6b_evaluate_steering trait_id=19 eval_trait_id=24
"""
import os, json, re, glob, random
import hydra
from omegaconf import DictConfig

from src.utils.paths import ensure_scratch_env
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.prompts_template import PROMPTS
from src.utils.judge_utils import compare_trait_pairwise_with_llm_judge
from src.utils.data_utils import load_trait_meta



def _find_strength_files(trait_dir: str, only=None, prefix: str = "strength"):
    """[(value, tag, path)] present on disk, optionally restricted to `only`.

    `only` is a list of values (numbers or strings); compared numerically so "0.10" and 0.1 match.
    `prefix` is the filename stem: "strength" for 5a_steer_trait output, "angle" for 5b_steer_angular output.
    """
    wanted = None if only is None else [float(x) for x in only]
    strength_files = []
    for path in glob.glob(os.path.join(trait_dir, f"{prefix}_*.json")):
        m = re.match(rf"^{re.escape(prefix)}_(-?\d*\.?\d+)\.json$", os.path.basename(path))
        if not m:
            continue
        s = float(m.group(1))
        if wanted is not None and not any(abs(s - w) < 1e-12 for w in wanted):
            continue
        strength_files.append((s, m.group(1), path))
    strength_files.sort(key=lambda x: x[0])
    return strength_files


def _judge_eval_trait(eval_id, strength_files, baseline, judge_model, judge_tokenizer,
                      out_dir, existing_summary, cfg):
    """Judge every requested strength for one eval trait. Returns the merged summary dict."""
    meta = load_trait_meta(eval_id)
    sign = str(cfg.get("eval_sign", "pos"))
    eval_trait_name = meta[f"trait_name_{sign}"]
    eval_trait_description = meta[f"base_description_{sign}"]
    summary = dict(existing_summary)

    for strength, tag, path in strength_files:
        with open(path, "r", encoding="utf-8") as f:
            steered = json.load(f)

        n = min(len(baseline), len(steered))
        if cfg.num_pairs:
            n = min(n, int(cfg.num_pairs))

        results = []
        n_steered = n_baseline = n_none = 0

        for i in tqdm(range(n), desc=f"eval_trait={eval_id} strength={tag}", leave=False):
            question = baseline[i]["question"]
            baseline_answer = baseline[i]["completion"]
            steered_answer = steered[i]["completion"]

            # The seed includes eval_id but not eval_sign, so the negative-pole run sees the same
            # answer pairs in the same A/B order as the positive run; only the trait named differs.
            rng = random.Random(f"{cfg.seed}-{cfg.trait_id}-{eval_id}-{tag}-{i}")
            a_is_steered = rng.random() < 0.5
            answer_a, answer_b = (steered_answer, baseline_answer) if a_is_steered else (baseline_answer, steered_answer)

            prompt = PROMPTS["trait_pairwise_v1"].format(
                trait_name=eval_trait_name,
                trait_description=eval_trait_description,
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
            )

            verdict = compare_trait_pairwise_with_llm_judge(
                model=judge_model,
                tokenizer=judge_tokenizer,
                prompt=prompt,
                max_retries=cfg.judge_max_retries,
            )

            if verdict == "A":
                winner = "steered" if a_is_steered else "baseline"
            elif verdict == "B":
                winner = "baseline" if a_is_steered else "steered"
            else:
                winner = None

            if winner == "steered":
                n_steered += 1
            elif winner == "baseline":
                n_baseline += 1
            else:
                n_none += 1

            results.append({"question": question, "winner": winner})

        decided = n_steered + n_baseline
        rate = (n_steered / decided) if decided else None

        with open(os.path.join(out_dir, f"{str(cfg.get('file_prefix','strength'))}_{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        summary[tag] = {
            "strength": strength,
            "n_pairs": n,
            "n_steered_preferred": n_steered,
            "n_baseline_preferred": n_baseline,
            "n_unparseable": n_none,
            "steered_preferred_rate": rate,
        }
        rate_str = f"{rate * 100:.1f}%" if rate is not None else "n/a"
        print(f"[INFO] eval_trait={eval_id} strength={tag}: steered preferred {n_steered}/{decided} "
              f"({rate_str}), {n_none} unparseable")

    return summary


# The Hydra configs interpolate ${oc.env:SCRATCH}, which OmegaConf resolves when the
# decorator below loads the config. Resolving it here first turns an unset SCRATCH into
# the clear error of utils/paths.py.
ensure_scratch_env()


@hydra.main(config_path="../configs", config_name="evaluate_steering", version_base=None)
def main(cfg: DictConfig) -> None:
    gen_meta = load_trait_meta(int(cfg.trait_id))

    # eval_trait_id may be a scalar or a list; with a list, the judge model is loaded only once for
    # all eval traits.
    raw = cfg.eval_trait_id
    eval_ids = [int(raw)] if isinstance(raw, (int, str)) else [int(x) for x in raw]

    # `input_dir` lets a differently-shaped producer (e.g. 5b_steer_angular's pair_<A>_<B>/) reuse
    # this judge path unchanged; `file_prefix` switches strength_*.json -> angle_*.json.
    prefix = str(cfg.get("file_prefix", "strength"))
    trait_dir = cfg.get("input_dir", None) or os.path.join(
        cfg.steering_dir, cfg.checkpoint, f"l1_{cfg.l1}", f"trait_{cfg.trait_id}")

    only = cfg.get("strengths", None)
    all_strengths = _find_strength_files(trait_dir, list(only) if only is not None else None, prefix)
    if not all_strengths:
        raise FileNotFoundError(
            f"No matching {prefix}_*.json in {trait_dir}"
            + (f" for strengths={list(only)}" if only is not None else "")
        )

    # Plan the work before loading the judge, so a run with nothing left to do does not load it.
    plan = []
    for eid in eval_ids:
        sign = str(cfg.get("eval_sign", "pos"))
        suffix = "" if sign == "pos" else f"_{sign}"
        base = cfg.get("output_dir", None) or os.path.join(
            cfg.out_dir, cfg.checkpoint, f"l1_{cfg.l1}", f"trait_{cfg.trait_id}")
        out_dir = os.path.join(base, f"eval_trait_{eid}{suffix}")
        summary_path = os.path.join(out_dir, "summary.json")
        existing = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        todo = all_strengths
        if cfg.get("skip_existing", False):
            todo = [sf for sf in all_strengths
                    if not (sf[1] in existing and os.path.exists(os.path.join(out_dir, f"{prefix}_{sf[1]}.json")))]
        if todo:
            plan.append((eid, out_dir, summary_path, existing, todo))

    if not plan:
        print("[INFO] nothing to judge; judge model not loaded.")
        return

    print(f"[INFO] gen_trait={cfg.trait_id} ({gen_meta['trait_name_pos']} vs {gen_meta['trait_name_neg']}) | "
          f"eval_sign={cfg.get('eval_sign','pos')} | "
          f"{len(plan)} eval trait(s) to judge | strengths={[s[1] for s in all_strengths]}")

    with open(os.path.join(trait_dir, "baseline.json"), "r", encoding="utf-8") as f:
        baseline = json.load(f)

    print(f"[INFO] Loading judge model: {cfg.judge_model}")
    judge_model = AutoModelForCausalLM.from_pretrained(cfg.judge_model, device_map="auto", torch_dtype=torch.float16).eval()
    judge_tokenizer = AutoTokenizer.from_pretrained(cfg.judge_model)
    judge_model.generation_config.temperature = None
    judge_model.generation_config.top_p = None
    judge_model.generation_config.top_k = None
    print("[INFO] Judge model loaded and ready.")

    for eid, out_dir, summary_path, existing, todo in plan:
        os.makedirs(out_dir, exist_ok=True)
        summary = _judge_eval_trait(eid, todo, baseline, judge_model, judge_tokenizer,
                                    out_dir, existing, cfg)
        summary = dict(sorted(summary.items(), key=lambda kv: float(kv[1]["strength"])))
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] wrote summary ({len(summary)} strengths): {summary_path}")


if __name__ == "__main__":
    main()
