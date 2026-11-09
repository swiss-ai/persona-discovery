"""Generate responses under a trait's positive and negative instructions and cache their activations.

For one trait and checkpoint, every question of each set in `sets` (default `cal`) is answered once
per instruction and sign. The completions and the pooled activations of every requested layer,
submodule and pooling mode are written in batches to
$SCRATCH/apertus-pulse/data/cache/<checkpoint>/trait_<id>/<set>/{completions,acts}/.
Batches whose activation files already exist are skipped, so an interrupted run can be resumed.

Usage (Hydra multirun for several traits):
    python -m src.2_prepare_probing_dataset trait_id=1,2,3 model.checkpoint=apertus_v1.5_8B --multirun
"""
import hydra
from omegaconf import DictConfig

from src.utils.paths import ensure_scratch_env

import os, json, torch, time
import gc
from tqdm import tqdm
from src.utils.data_utils import load_trait_instructions, build_prompts, load_questions
from src.utils.model_utils import activation_pooling, load_model_and_tokenizer, generate_and_capture_activations, get_num_hidden_layers


def cache_activations(
    trait_id,
    set_name,
    model,
    tokenizer,
    set_dir,
    batch_size,
    max_new_tokens,
    temperature,
    pooling_modes,
    layers,
    submodules,
) -> None:

    # Load trait instructions, questions and layers to save activations from.
    trait_instructions = load_trait_instructions(trait_id)
    questions = load_questions(set_name)
    layers = list(range(get_num_hidden_layers(model))) if layers is None else [int(x) for x in str(layers).split(",")]

    for sign in ["pos", "neg"]:

        # Build all prompts for the sign.
        all_prompts = build_prompts(trait_instructions, questions, sign)

        for i in tqdm(range(0, len(all_prompts), batch_size), desc=f"{set_name}"):
            
            # Skip the batch if all its activation files exist (assumes that its completions were
            # written as well; they are not checked).
            acts_exist = True
            for layer_id in layers:
                for sub in submodules:
                    for pool in pooling_modes:
                        act_path = os.path.join(set_dir, f"acts/layer_{layer_id}", sub, f"{pool}", f"{sign}_batch_{i}.pt")
                        if not os.path.exists(act_path):
                            acts_exist = False
            if acts_exist:
                print(f"[INFO] Skipping batch {i} for sign '{sign}' as activations already exist.")
                continue

            batch = all_prompts[i : i + batch_size]
            prompts = [p["prompt"] for p in batch]
            
            gen_ids, gen_spans, store = generate_and_capture_activations(model, tokenizer, prompts, layers, submodules, max_new_tokens, temperature)
            
            texts = []
            if tokenizer.chat_template is not None:
                print(f"[DEBUG] Using tokenizer's chat template for formatting prompts.")
                texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
            else:
                print(f"[DEBUG] No chat template found in tokenizer, manually formatting prompts.")
                # Fallback: manually format the chat messages
                for p in prompts:
                    text = ""
                    for msg in p:
                        if msg["role"] == "system":
                            text += f"{msg['content']}\n\n"
                        elif msg["role"] == "user":
                            text += f"User: {msg['content']}\n\nAssistant:"
                    text += " "
                    texts.append(text)
        
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
            gen_texts = tokenizer.batch_decode(gen_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
            
            completions = []
            for index, (meta, ans) in enumerate(zip(batch, gen_texts)):
                completions.append({
                    "index": index + i,
                    "instruction": meta["prompt"][0]["content"],
                    "question": meta["question"],
                    "completion": ans,
                    "label": meta["label"],
                    "trait_sign": meta["trait_sign"]
                })

            os.makedirs(f"{set_dir}/completions", exist_ok=True)
            with open(f"{set_dir}/completions/{meta['trait_sign']}_completions_batch_{i}.json", "w") as f:
                json.dump(completions, f, indent=2, ensure_ascii=False)
                    
            # Save activations per batch.
            printed_debug = False
            pooled_acc = {} 
            for (layer_id, sub), acts in store.items():
                for j, span in enumerate(gen_spans):
                    for pool in pooling_modes:
                        key = (layer_id, sub, pool, batch[j]["trait_sign"]) # include sign to separate outputs
                        pooled = activation_pooling(acts[j], span, pool)
                        pooled_acc.setdefault(key, []).append(pooled.cpu())

                    if i == 0 and j == 0 and layer_id == layers[0] and not printed_debug:
                        gen_start, gen_end = span
                        print(f"[DEBUG] Batch 0, example 0")
                        print(f"[DEBUG] Layer: {layer_id}, Submodule: {sub}")
                        print(f"[DEBUG] Acts shape: {acts[j].shape}")  # [seq_len, hidden_dim]
                        print(f"[DEBUG] Generated span (padded coords): {gen_start}:{gen_end} ({gen_end - gen_start} tokens)")
                        print(f"[DEBUG] Full prompt:\n{batch[j]['prompt']}")
                        print(f"[DEBUG] Generated completion:\n{gen_texts[j]}")
                        print(f"[DEBUG] Token norms at prompt/gen boundary:",
                            f"prompt={acts[j][gen_start - 1].norm().item():.4f},",
                            f"gen={acts[j][gen_start].norm().item():.4f}" if gen_start < acts[j].shape[0] else "N/A")
                        print(f"[DEBUG] First 5 tokens prompt:", acts[j][:gen_start].norm(dim=1)[:5].tolist())
                        print(f"[DEBUG] First 5 tokens gen:", acts[j][gen_start:gen_end].norm(dim=1)[:5].tolist() if gen_end > gen_start else "N/A")
                        print("")
                        printed_debug = True
            
            for (layer_id, sub, pool, sign), tensors in pooled_acc.items():
                save_dir = os.path.join(set_dir, f"acts/layer_{layer_id}", sub, f"{pool}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{sign}_batch_{i}.pt")
                try:
                    torch.save(torch.stack(tensors, dim=0), save_path)
                except RuntimeError:  # tensors of different shapes cannot be stacked: save the list instead
                    torch.save(tensors, save_path)  

            try:
                del acts, tensors, pooled
            except NameError:
                pass
            del gen_ids, inputs, store, gen_spans
            del completions, batch, prompts, texts, gen_texts
            del pooled_acc
            gc.collect()
            torch.cuda.empty_cache()

    return 

# The Hydra configs interpolate ${oc.env:SCRATCH}, which OmegaConf resolves when the
# decorator below loads the config. Resolving it here first turns an unset SCRATCH into
# the clear error of utils/paths.py.
ensure_scratch_env()


@hydra.main(config_path="../configs", config_name="prepare_probing_dataset", version_base=None)
def main(cfg: DictConfig) -> None:

    model, tokenizer = load_model_and_tokenizer(cfg)

    total_start = time.time()
    for set_name in list(cfg.sets):
        set_start = time.time()
        set_dir = os.path.join(cfg.data_dir, cfg.model.checkpoint, f"trait_{cfg.trait_id}", set_name)
        os.makedirs(set_dir, exist_ok=True)

        print(f"[INFO] Caching activations for set '{set_name}' using model checkpoint '{cfg.model.checkpoint}'")
        cache_activations(
            trait_id=cfg.trait_id,
            set_name=set_name,
            model=model,
            tokenizer=tokenizer,
            set_dir=set_dir,
            batch_size=cfg.batch_size,
            layers=cfg.layers,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            pooling_modes=cfg.pooling_modes,
            submodules=cfg.submodules,
        )
        print(f"[TIMING] cache_activations({set_name}): {time.time() - set_start:.2f}s")
    
    total_elapsed = time.time() - total_start
    pooling_count = len(cfg.pooling_modes) if cfg.pooling_modes is not None else 0
    submodules_count = len(cfg.submodules) if cfg.submodules is not None else 0
    print(f"[TIMING] for {len(cfg.sets)} sets for {cfg.model.checkpoint} with {pooling_count} pooling modes and {submodules_count} submodules it took {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()

