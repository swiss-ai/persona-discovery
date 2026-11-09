"""LLM-judge call for the pairwise steering evaluation (src/6b_evaluate_steering.py)."""
from typing import Optional
import re
import torch


def compare_trait_pairwise_with_llm_judge(model, tokenizer, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Ask the judge which answer slot better exhibits the trait.

    `prompt` is the complete pairwise prompt with both answers already in their slots. Decodes greedily
    and returns "A" or "B", or None if no answer can be parsed or generation fails `max_retries` times.
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    patterns = [
                r'^\s*([AB])\s*$',      # Just a letter
                r'\b([AB])\s*$',        # Letter at end
                r'answer[:\s]+([AB])',  # "answer: A" or "answer A"
                r'\b([AB])\b',          # Any A/B with word boundaries
            ]

    for attempt in range(max_retries):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    return match.group(1).upper()

            # No clean A/B answer found
            print(f"[WARN] Failed to parse pairwise judge response: {response}")
            return None

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WARN] Error during pairwise evaluation: {e}, retrying...")
            else:
                print(f"[ERROR] Failed after {max_retries} attempts: {e}")
                return None

