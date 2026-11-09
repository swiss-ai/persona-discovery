"""Download the PsychoCounsel-Preference questions and split them into train, cal and test sets.

The dataset's train split is divided into `train` and `cal` (fraction `cal_ratio`, seed 42); its test
split becomes `test`. Duplicates are removed within each split. Output:
$SCRATCH/apertus-pulse/data/questions/{train,cal,test}.json. The probes use `cal`, steering uses `test`.

    python -m src.0_dump_questions
"""
import os
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.utils.paths import data_path


def save_jsonl(path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"questions": data}, f, ensure_ascii=False, indent=2)


def dedupe(items: list) -> list:
    """Order-preserving de-duplication (dict keys keep insertion order).

    PsychoCounsel-Preference is a *preference* dataset: the same question appears in several
    rows as different chosen/rejected pairs (34,329 rows but only 15,076 distinct questions).
    Splitting the raw row list therefore yields repeated questions within a split, which would
    spend generation budget re-answering the same prompt. De-duplicating after the split keeps
    the split identity itself unchanged.
    """
    return list(dict.fromkeys(items))


def main(cal_ratio=0.015, output_dir=None):
    if output_dir is None:
        output_dir = data_path("questions")
    
    ds = load_dataset("Psychotherapy-LLM/PsychoCounsel-Preference")
    train_data = [i["question"] for i in ds["train"]]
    test_data = [i["question"] for i in ds["test"]]
    train_examples, cal_examples = train_test_split(
        train_data, test_size=cal_ratio, random_state=42, shuffle=True
    )

    # De-duplicate within each split (the split itself is left untouched).
    train_examples, cal_examples, test_data = dedupe(train_examples), dedupe(cal_examples), dedupe(test_data)
    print(f"[INFO] cal_ratio={cal_ratio} | distinct questions: train={len(train_examples)} cal={len(cal_examples)} test={len(test_data)}")

    os.makedirs(output_dir, exist_ok=True)
    save_jsonl(os.path.join(output_dir, "train.json"), train_examples)
    save_jsonl(os.path.join(output_dir, "cal.json"), cal_examples)
    save_jsonl(os.path.join(output_dir, "test.json"), test_data)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
