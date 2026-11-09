"""Generate positive and negative trait descriptions with an LLM.

Reads the trait names and reference descriptions in data/traits/pos_traits_descriptions.json, asks
`prompt_model` (prompt `generate_pos_neg_traits_v2`) for a positive and a negative description of each
trait, and writes data/traits/all_traits.json. With --trait_id only that trait is generated, and
all_traits.json then contains only that trait. The shipped data/traits/ already contains the result.

    python -m src.1a_prepare_traits
"""
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts_template import PROMPTS
from src.utils.paths import repo_path


def extract_json(text: str):
    # Strategy 1: Look for <json> ... </json> tags.
    if "<json>" in text and "</json>" in text:
        snippet = text.split("<json>", 1)[1].split("</json>", 1)[0]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    # Strategy 2: Look for JSON code blocks (```json ... ```).
    if "```json" in text:
        snippet = text.split("```json", 1)[1].split("```", 1)[0]
        try:
            return json.loads(snippet.strip())
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find the first { and last } and try to parse that.
    try:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            snippet = text[start_idx : end_idx + 1]
            return json.loads(snippet)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try the whole text as JSON.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


def main(
    prompt_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
    output_dir=None,
    trait_id: int = None,
):
    output_dir = output_dir or repo_path("data", "traits")
    os.makedirs(output_dir, exist_ok=True)

    with open(repo_path("data", "traits", "pos_traits_descriptions.json"), "r") as f:
        traits = json.load(f)["therapist_traits_with_descriptions"]
    if trait_id is not None:
        traits = [t for t in traits if t["trait_id"] == trait_id]

    tokenizer = AutoTokenizer.from_pretrained(prompt_model)
    model = AutoModelForCausalLM.from_pretrained(prompt_model, device_map="auto")

    all_results = []
    for t in traits:
        prompt = PROMPTS["generate_pos_neg_traits_v2"].format(
            trait_id=t["trait_id"],
            trait_name_pos=t["trait_name_pos"],
            trait_description_pos=t["trait_description_pos"],
        )

        result = None
        for attempt in range(1, 4):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, max_new_tokens=2000, temperature=0.7, do_sample=True
            )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            )

            result = extract_json(response)
            if result:
                if attempt > 1:
                    print(f"trait {t['trait_id']} attempt {attempt}")
                break
            print(f"trait {t['trait_id']} attempt {attempt} failed")
            print(
                f"ERROR: Could not extract JSON from response for trait: {t['trait_id']} with {attempt} attempts"
            )
            print(f"[DEBUG] Full response:\n{response}\n")

        if not result:
            print(f"ERROR: trait {t['trait_id']} failed after 3 attempts.")
            continue

        result["trait_id"] = t["trait_id"]
        all_results.append(result)
        print(f"trait {t['trait_id']} succeeded.")


    path = os.path.join(
        output_dir,
        f"all_traits.json",
    )
    with open(path, "w") as f:
        json.dump({"therapist_traits": all_results}, f, indent=2, ensure_ascii=False)
    print(f"Saved {path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
