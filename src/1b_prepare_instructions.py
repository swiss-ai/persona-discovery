"""Generate the instruction pairs of each trait with an LLM.

Reads data/traits/all_traits.json, asks `prompt_model` (prompt `generate_instructions_v3`) for
`nr_instructions` pairs of system instructions per trait (one expressing the trait, one its negative
pole), and writes data/traits/<id>_<pos>_vs_<neg>.json. The shipped files contain three pairs per trait
(`--nr_instructions 3`); the default is 2.

    python -m src.1b_prepare_instructions --nr_instructions 3
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
    nr_instructions: int = 2,
    trait_id: int = None,
):
    output_dir = output_dir or repo_path("data", "traits")
    os.makedirs(output_dir, exist_ok=True)

    with open(repo_path("data", "traits", "all_traits.json"), "r") as f:
        traits = json.load(f)["therapist_traits"]
    if trait_id is not None:
        traits = [t for t in traits if t["trait_id"] == trait_id]

    tokenizer = AutoTokenizer.from_pretrained(prompt_model)
    model = AutoModelForCausalLM.from_pretrained(prompt_model, device_map="auto")

    for t in traits:
        prompt = PROMPTS["generate_instructions_v3"].format(
            trait_id=t["trait_id"],
            trait_name_pos=t["trait_name_pos"],
            trait_name_neg=t["trait_name_neg"],
            base_description_pos=t["base_description_pos"],
            base_description_neg=t["base_description_neg"],
            nr_instructions=nr_instructions,
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

        path = os.path.join(
            output_dir,
            f"{t['trait_id']}_{t['trait_name_pos']}_vs_{t['trait_name_neg']}.json",
        )
        with open(path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved {path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
