"""Generate unsteered and steered answers for one trait.

Answers the first `num_questions` questions of the `test` set once without steering (baseline) and
once per value in `strengths`. Steering adds the trait's probe direction (probes trained with penalty
`l1`) at every layer that has a probe, scaled relative to each token's activation norm (see
utils.model_utils.register_steering_hook). Each completion is also scored with the same probes; these
self-scores are a diagnostic, not independent evidence of steering, because they read out the
direction that was added. Output:
$SCRATCH/apertus-pulse/data/steering/<checkpoint>/l1_<l1>/trait_<id>/{baseline,strength_<s>}.json.

Usage:
    python -m src.5a_steer_trait trait_id=1 model.checkpoint=apertus_v1.5_8B probe_checkpoint=apertus_v1.5_8B
"""
import os, json
import hydra
from omegaconf import DictConfig

from src.utils.paths import ensure_scratch_env
import torch
from tqdm import tqdm

from src.utils.data_utils import load_questions, build_default_prompts
from src.utils.model_utils import (
    load_model_and_tokenizer,
    load_probes,
    get_transformer_layers,
    get_num_hidden_layers,
    generate_and_capture_activations,
    activation_pooling,
    register_steering_hook,
)


def run_condition(model, tokenizer, prompts, probes, submodule, pooling, strength, batch_size, max_new_tokens):
    transformer_layers = get_transformer_layers(model)
    layers = sorted(probes.keys())

    handles = []
    if strength != 0.0:
        for layer_id in layers:
            direction = probes[layer_id].w.squeeze().detach().float()
            handles += register_steering_hook(transformer_layers[layer_id], layer_id, submodule, direction, strength)

    completions = []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"strength={strength}"):
            batch = prompts[i : i + batch_size]
            batch_prompts = [p["prompt"] for p in batch]

            gen_ids, gen_spans, store = generate_and_capture_activations(
                model, tokenizer, batch_prompts, layers, [submodule], max_new_tokens, 0.0
            )

            # Decoding and pooling both use the token span returned above.
            gen_start = gen_spans[0][0]
            gen_texts = tokenizer.batch_decode(gen_ids[:, gen_start:], skip_special_tokens=True)

            for j, (meta, ans) in enumerate(zip(batch, gen_texts)):
                probe_scores = {}
                for layer_id in layers:
                    acts = store.get((layer_id, submodule))
                    if acts is None:
                        continue
                    pooled = activation_pooling(acts[j], gen_spans[j], pooling)
                    with torch.no_grad():
                        score = probes[layer_id](pooled.cuda().float().unsqueeze(0)).sigmoid().item()
                    probe_scores[layer_id] = score

                completions.append({
                    "question": meta["question"],
                    "completion": ans,
                    "probe_scores": probe_scores,
                })

            del gen_ids, store
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    return completions


# The Hydra configs interpolate ${oc.env:SCRATCH}, which OmegaConf resolves when the
# decorator below loads the config. Resolving it here first turns an unset SCRATCH into
# the clear error of utils/paths.py.
ensure_scratch_env()


@hydra.main(config_path="../configs", config_name="steer_trait", version_base=None)
def main(cfg: DictConfig) -> None:
    model, tokenizer = load_model_and_tokenizer(cfg)

    layers = list(range(get_num_hidden_layers(model))) if cfg.layers is None else [int(x) for x in str(cfg.layers).split(",")]

    probes = load_probes(cfg.probe_dir, cfg.probe_checkpoint, cfg.trait_id, cfg.trained_set, cfg.submodule, cfg.pooling, cfg.l1, layers)
    if not probes:
        raise FileNotFoundError(
            f"No probes loaded for trait_id={cfg.trait_id} checkpoint={cfg.probe_checkpoint} "
            f"set={cfg.trained_set} submodule={cfg.submodule} pooling={cfg.pooling} l1={cfg.l1}. "
            "Check these match a trained 3_train_probes run."
        )
    if len(probes) < len(layers):
        missing = sorted(set(layers) - set(probes.keys()))
        print(f"[WARN] Missing probes for layers {missing}; steering will only apply at {sorted(probes.keys())}")

    questions = load_questions(cfg.test_set)
    if cfg.num_questions:
        questions = questions[: cfg.num_questions]
    prompts = build_default_prompts(questions, system_prompt=cfg.system_prompt)

    # l1 is part of the path: each penalty gives a different probe direction, so runs with
    # different l1 must not overwrite each other.
    out_dir = os.path.join(cfg.out_dir, cfg.model.checkpoint, f"l1_{cfg.l1}", f"trait_{cfg.trait_id}")
    os.makedirs(out_dir, exist_ok=True)

    conditions = [(0.0, "baseline")] + [(float(s), f"strength_{s}") for s in cfg.strengths]

    for strength, tag in conditions:
        completions = run_condition(
            model, tokenizer, prompts, probes, cfg.submodule, cfg.pooling, strength, cfg.batch_size, cfg.max_new_tokens
        )

        out_path = os.path.join(out_dir, f"{tag}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(completions, f, indent=2, ensure_ascii=False)

        all_scores = [s for c in completions for s in c["probe_scores"].values()]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else float("nan")
        print(f"[INFO] {tag}: wrote {len(completions)} completions to {out_path} | avg probe score={avg_score:.3f}")


if __name__ == "__main__":
    main()
