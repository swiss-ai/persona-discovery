"""Steer along mixtures of two traits' probe directions.

For a pair of traits A and B, the direction at each layer with probes for both traits is

    d(theta) = cos(theta) * unit(w_A) + sin(theta) * unit(w_B)

for each mixing value theta in `angles` (degrees), applied at a single `strength`. Both probe vectors
are normalised before mixing, because their raw norms differ across layers and traits and would
otherwise weight the mixture; register_steering_hook normalises d(theta) again and steers relative to
each token's activation norm, so the perturbation has the same size at every theta.

theta = 0 is pure A and theta = 90 pure B at every layer, which makes these two conditions directly
comparable with single-trait steering. theta is therefore a mixing parameter, not a geometric angle:
the actual angle between A and B is arccos(cosine) and varies by layer. (An orthonormalised basis
would rotate uniformly but would not put pure B at theta = 90.) The per-layer cosines are saved in
meta.json. Output: $SCRATCH/apertus-pulse/data/steering_angular/<checkpoint>/l1_<l1>/pair_<A>_<B>/.

Usage:
    python -m src.5b_steer_angular trait_a=7 trait_b=24 model.checkpoint=apertus_v1.5_8B
"""
import math
import os, json
import hydra
from omegaconf import DictConfig

from src.utils.paths import ensure_scratch_env
import torch
from tqdm import tqdm

from src.utils.data_utils import load_questions, build_default_prompts, load_trait_meta
from src.utils.model_utils import (
    load_model_and_tokenizer,
    load_probes,
    get_transformer_layers,
    get_num_hidden_layers,
    generate_and_capture_activations,
    activation_pooling,
    register_steering_hook,
)



def combined_directions(probes_a, probes_b, theta_deg):
    """{layer: cos(t)*unit(w_A) + sin(t)*unit(w_B)} over layers present in both probe sets."""
    t = math.radians(theta_deg)
    ca, sa = math.cos(t), math.sin(t)
    out = {}
    for layer in sorted(set(probes_a) & set(probes_b)):
        wa = probes_a[layer].w.squeeze().detach().float()
        wb = probes_b[layer].w.squeeze().detach().float()
        out[layer] = ca * (wa / wa.norm()) + sa * (wb / wb.norm())
    return out


def run_angle(model, tokenizer, prompts, directions, probes_a, probes_b, submodule, pooling,
              strength, batch_size, max_new_tokens, tag):
    """Generate under one combined direction. Self-scores with both traits' probes (diagnostic only)."""
    transformer_layers = get_transformer_layers(model)
    layers = sorted(directions.keys())

    handles = []
    if strength != 0.0:
        for layer_id in layers:
            handles += register_steering_hook(
                transformer_layers[layer_id], layer_id, submodule, directions[layer_id], strength
            )

    completions = []
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=tag):
            batch = prompts[i : i + batch_size]
            batch_prompts = [p["prompt"] for p in batch]

            gen_ids, gen_spans, store = generate_and_capture_activations(
                model, tokenizer, batch_prompts, layers, [submodule], max_new_tokens, 0.0
            )
            gen_start = gen_spans[0][0]
            gen_texts = tokenizer.batch_decode(gen_ids[:, gen_start:], skip_special_tokens=True)

            for j, (meta, ans) in enumerate(zip(batch, gen_texts)):
                scores_a, scores_b = {}, {}
                for layer_id in layers:
                    acts = store.get((layer_id, submodule))
                    if acts is None:
                        continue
                    pooled = activation_pooling(acts[j], gen_spans[j], pooling)
                    with torch.no_grad():
                        x = pooled.cuda().float().unsqueeze(0)
                        scores_a[layer_id] = probes_a[layer_id](x).sigmoid().item()
                        scores_b[layer_id] = probes_b[layer_id](x).sigmoid().item()

                completions.append({
                    "question": meta["question"],
                    "completion": ans,
                    "probe_scores_a": scores_a,
                    "probe_scores_b": scores_b,
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


@hydra.main(config_path="../configs", config_name="steer_angular", version_base=None)
def main(cfg: DictConfig) -> None:
    a, b = int(cfg.trait_a), int(cfg.trait_b)
    meta_a, meta_b = load_trait_meta(a), load_trait_meta(b)
    model, tokenizer = load_model_and_tokenizer(cfg)

    layers = list(range(get_num_hidden_layers(model))) if cfg.layers is None else [int(x) for x in str(cfg.layers).split(",")]
    kw = dict(set_name=cfg.trained_set, submodule=cfg.submodule, pooling=cfg.pooling, l1=cfg.l1, layers=layers)
    probes_a = load_probes(cfg.probe_dir, cfg.probe_checkpoint, a, **kw)
    probes_b = load_probes(cfg.probe_dir, cfg.probe_checkpoint, b, **kw)
    shared = sorted(set(probes_a) & set(probes_b))
    if not shared:
        raise FileNotFoundError(f"No layers with probes for BOTH trait {a} and trait {b} at l1={cfg.l1}")
    print(f"[INFO] pair {a} ({meta_a['trait_name_pos']}) + {b} ({meta_b['trait_name_pos']}) | "
          f"{len(shared)} shared layers | strength={cfg.strength}")

    # Per-layer cosine between the two directions: context for interpreting theta, since theta is a
    # mixing parameter and the true A->B arc is arccos(c), which varies by layer.
    cos_by_layer = {}
    for L in shared:
        wa = probes_a[L].w.squeeze().detach().float(); wb = probes_b[L].w.squeeze().detach().float()
        cos_by_layer[L] = float((wa @ wb / (wa.norm() * wb.norm())).item())
    cs = list(cos_by_layer.values())
    print(f"[INFO] cos(A,B) per layer: mean={sum(cs)/len(cs):+.3f} min={min(cs):+.3f} max={max(cs):+.3f} "
          f"-> true A->B angle spans {math.degrees(math.acos(max(cs))):.0f}-{math.degrees(math.acos(min(cs))):.0f} deg")

    questions = load_questions(cfg.test_set)
    if cfg.num_questions:
        questions = questions[: cfg.num_questions]
    prompts = build_default_prompts(questions, system_prompt=cfg.system_prompt)

    out_dir = os.path.join(cfg.out_dir, cfg.model.checkpoint, f"l1_{cfg.l1}", f"pair_{a}_{b}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "trait_a": a, "trait_b": b,
            "name_a": meta_a["trait_name_pos"], "name_b": meta_b["trait_name_pos"],
            "checkpoint": cfg.model.checkpoint, "l1": cfg.l1, "strength": cfg.strength,
            "angles": list(cfg.angles), "num_questions": len(questions),
            "cos_by_layer": cos_by_layer,
            "note": "d(theta)=cos(theta)*unit(w_A)+sin(theta)*unit(w_B) per layer. theta is a mixing "
                    "parameter, not a true angle: the real A->B arc is arccos(cos) and varies by layer. "
                    "theta=0 is exactly pure A and theta=90 exactly pure B at every layer.",
        }, f, indent=2)

    conditions = [(None, "baseline")] + [(float(t), f"angle_{t}") for t in cfg.angles]
    for theta, tag in conditions:
        if theta is None:
            directions = {L: probes_a[L].w.squeeze().detach().float() for L in shared}  # unused at strength 0
            strength = 0.0
        else:
            directions = combined_directions(probes_a, probes_b, theta)
            strength = float(cfg.strength)

        completions = run_angle(model, tokenizer, prompts, directions, probes_a, probes_b,
                                cfg.submodule, cfg.pooling, strength, cfg.batch_size,
                                cfg.max_new_tokens, tag)

        with open(os.path.join(out_dir, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(completions, f, indent=2, ensure_ascii=False)

        sa = [s for c in completions for s in c["probe_scores_a"].values()]
        sb = [s for c in completions for s in c["probe_scores_b"].values()]
        print(f"[INFO] {tag}: wrote {len(completions)} completions | "
              f"avg probe A={sum(sa)/len(sa):.3f} B={sum(sb)/len(sb):.3f}")


if __name__ == "__main__":
    main()
