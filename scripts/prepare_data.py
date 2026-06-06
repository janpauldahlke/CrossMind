"""
Phase 1.2: Download and prepare the public alignment dataset.

Sources:
  - tatsu-lab/alpaca: general instruction-following (2500 samples)
  - qiaojin/PubMedQA (pqa_artificial): medical research Q&A (1000 samples)
  - lavita/ChatDoctor-HealthCareMagic-100k: doctor-patient dialogues (500 samples)

Total: 4000 samples, saved to data/public_dataset/texts.jsonl
Validation split: 500 samples (sampled evenly from each source).
"""

import json
import random
import sys
from pathlib import Path

from datasets import load_dataset


def fetch_alpaca(n: int = 2500) -> list[dict]:
    """Fetch Alpaca samples, concatenating instruction + input + output."""
    print(f"  Loading tatsu-lab/alpaca ({n} samples)...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    samples = []
    for row in ds:
        parts = [row["instruction"]]
        if row.get("input", "").strip():
            parts.append(row["input"])
        parts.append(row["output"])
        text = "\n\n".join(parts)
        if len(text) > 50:
            samples.append({"text": text, "source": "alpaca"})
        if len(samples) >= n:
            break
    return samples


def fetch_pubmedqa(n: int = 1000) -> list[dict]:
    """Fetch PubMedQA artificial split, combining question + long answer."""
    print(f"  Loading qiaojin/PubMedQA pqa_artificial ({n} samples)...")
    ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")

    samples = []
    for row in ds:
        question = row.get("question", "")
        long_answer = row.get("long_answer", "")
        text = f"Question: {question}\n\nAnswer: {long_answer}"
        if len(text) > 50:
            samples.append({"text": text, "source": "pubmedqa"})
        if len(samples) >= n:
            break
    return samples


def fetch_chatdoctor(n: int = 500) -> list[dict]:
    """Fetch ChatDoctor-HealthCareMagic dialogues."""
    print(f"  Loading lavita/ChatDoctor-HealthCareMagic-100k ({n} samples)...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")

    samples = []
    for row in ds:
        instruction = row.get("instruction", row.get("input", ""))
        output = row.get("output", row.get("response", ""))
        text = f"Patient: {instruction}\n\nDoctor: {output}"
        if len(text) > 50:
            samples.append({"text": text, "source": "chatdoctor"})
        if len(samples) >= n:
            break
    return samples


def main():
    random.seed(42)
    out_dir = Path("data/public_dataset")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "texts.jsonl"

    print("Preparing public alignment dataset...")

    all_samples = []

    try:
        all_samples.extend(fetch_alpaca(2500))
    except Exception as e:
        print(f"  WARNING: Failed to load Alpaca: {e}", file=sys.stderr)

    try:
        all_samples.extend(fetch_pubmedqa(1000))
    except Exception as e:
        print(f"  WARNING: Failed to load PubMedQA: {e}", file=sys.stderr)

    try:
        all_samples.extend(fetch_chatdoctor(500))
    except Exception as e:
        print(f"  WARNING: Failed to load ChatDoctor: {e}", file=sys.stderr)

    if len(all_samples) < 100:
        print(f"ERROR: Only {len(all_samples)} samples collected, need at least 100.")
        sys.exit(1)

    random.shuffle(all_samples)

    # Split: last 500 for validation, rest for training
    val_per_source = {}
    for s in all_samples:
        src = s["source"]
        val_per_source.setdefault(src, [])

    val_samples = []
    train_samples = []
    val_budget = {"alpaca": 200, "pubmedqa": 175, "chatdoctor": 125}

    for s in all_samples:
        src = s["source"]
        budget = val_budget.get(src, 0)
        if len(val_per_source[src]) < budget:
            val_per_source[src].append(s)
            s["split"] = "val"
            val_samples.append(s)
        else:
            s["split"] = "train"
            train_samples.append(s)

    with open(out_path, "w") as f:
        for sample in train_samples + val_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nDataset saved to {out_path}")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Total: {len(train_samples) + len(val_samples)}")

    source_counts = {}
    for s in all_samples:
        source_counts[s["source"]] = source_counts.get(s["source"], 0) + 1
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")


if __name__ == "__main__":
    main()
