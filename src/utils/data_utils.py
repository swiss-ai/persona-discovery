"""Trait definitions, instruction pairs, question splits and chat prompts for the GPU stages."""
import os, json

from src.utils.paths import repo_path, data_path



def load_trait_instructions(trait_id: int) -> dict:
    """Return the instruction file of a trait (data/traits/<id>_*.json) merged with its entry in all_traits.json."""
    traits_dir = repo_path("data", "traits")
    all_traits_path = os.path.join(traits_dir, "all_traits.json")

    with open(all_traits_path, "r", encoding="utf-8") as f:
        all_traits = json.load(f)["therapist_traits"]

    trait_meta = next((t for t in all_traits if t["trait_id"] == trait_id), None)
    if trait_meta:
        print(f"[DEBUG] Loaded trait: {trait_meta['trait_name_pos']} vs {trait_meta['trait_name_neg']}")
    if not trait_meta:
        raise FileNotFoundError(f"trait {trait_id} not found in {all_traits_path}")

    for fname in os.listdir(traits_dir):
        if fname.startswith(f"{trait_id}_") and fname.endswith(".json"):
            with open(os.path.join(traits_dir, fname), "r", encoding="utf-8") as f:
                trait_data = json.load(f)
                trait_data.update(trait_meta)
                return trait_data
    raise FileNotFoundError(f"trait {trait_id} file not found in {traits_dir}")
    
    
def load_trait_meta(trait_id: int) -> dict:
    all_traits_path = repo_path("data", "traits", "all_traits.json")

    with open(all_traits_path, "r", encoding="utf-8") as f:
        all_traits = json.load(f)["therapist_traits"]

    for t in all_traits:
        if int(t["trait_id"]) == int(trait_id):
            return t
    raise ValueError(f"trait_id={trait_id} not found in {all_traits_path}")


def load_questions(set_name):
    path = data_path("questions", f"{set_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No question split at {path}. The splits are deterministic -- regenerate with "
            "`python -m src.0_dump_questions` (fixed seed 42, cal_ratio 0.015, de-duplicated "
            "within each split)."
        )
    with open(path) as f:
        data = json.load(f)
    return data["questions"] if isinstance(data, dict) else data


def build_prompts(trait_data: dict, questions: list, trait_sign: str):
    prompts = []
    for instr in [x[trait_sign] for x in trait_data["instruction"]]:
        for q in questions:
            sys = f"You are a {trait_data['trait_name_' + trait_sign]} therapist. {instr}"
            prompts.append({
                "prompt": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": q}
                ],
                "question": q,
                "label": 1 if trait_sign == "pos" else 0,
                "trait_sign": trait_sign
            })
    return prompts


def build_default_prompts(questions, system_prompt=None):
    prompts = []
    for q in questions:
        if system_prompt:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q},
            ]
        else:
            prompt = [
                {"role": "user", "content": q},
            ]
        prompts.append({
            "prompt": prompt,
            "question": q,
        })
    return prompts
