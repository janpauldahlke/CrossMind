#!/usr/bin/env python3
"""Label medical texts into 5 routing categories using Qwen zero-shot classification.

Reads data/public_dataset/texts.jsonl, filters to medical/clinical samples,
classifies each into Cardiology / Neurology / Oncology / Orthopedics /
General_Medicine, and writes data/routing_labels.jsonl.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_config, load_model, unload_model

CATEGORIES = [
    (0, "Cardiology"),
    (1, "Neurology"),
    (2, "Oncology"),
    (3, "Orthopedics"),
    (4, "General_Medicine"),
]
CATEGORY_NAMES = [name for _, name in CATEGORIES]
NAME_TO_LABEL = {name.lower(): label for label, name in CATEGORIES}

MEDICAL_SOURCES = {"pubmedqa", "chatdoctor"}

STRONG_MEDICAL_KEYWORDS = [
    "patient", "symptom", "diagnosis", "disease", "clinical",
    "treatment", "medication", "hospital", "physician", "surgery",
    "cancer", "diabetes", "infection", "fever", "blood pressure",
    "prescription", "tumor", "biopsy", "chemotherapy", "stroke", "seizure",
    "fracture", "arthritis", "ecg", "troponin", "arrhythmia", "malignancy",
    "prognosis", "pathology", "radiology", "dosage", "antibiotic",
    "hypertension", "pregnancy", "obstetric", "pediatric", "diagnostic",
    "therapeutic", "syndrome", "pathogen", "anesthesia", "icu",
]

WEAK_MEDICAL_KEYWORDS = [
    "medical", "doctor", "nurse", "therapy", "drug", "renal", "hepatic",
    "pulmonary", "oncology", "neurology", "cardiology", "orthopedic",
]

NON_MEDICAL_HINTS = [
    "write a poem", "write a haiku", "write a story", "write a song",
    "photosynthesis", "bitcoin", "cryptocurrency", "meditation",
    "recipe", "camping", "vacation", "movie title", "reddit thread",
    "programming language", "python", "javascript", "haiku poem",
]

MEDICAL_CHECK_PROMPT = (
    "Is the following a medical or clinical patient-care query "
    "(symptoms, diagnosis, treatment, lab results, medications)?\n"
    "Answer NO for general wellness tips, fitness advice, news headlines, "
    "writing tasks, and non-clinical topics.\n"
    "Reply with ONLY YES or NO.\n\n"
    "Text: {text}\n\nAnswer:"
)

ROUTING_PROMPT = (
    "Classify this medical query into exactly one category. "
    "Reply with ONLY the category name.\n"
    "Categories: Cardiology, Neurology, Oncology, Orthopedics, General_Medicine\n"
    "Query: {text}\n"
    "Category:"
)

MAX_QUERY_CHARS = 2000


def log(msg: str) -> None:
    print(msg, flush=True)


def load_labeled_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                indices.add(json.loads(line)["sample_idx"])
    return indices


def load_samples(path: Path) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def truncate_text(text: str, max_chars: int = MAX_QUERY_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def extract_query_text(text: str, source: str) -> str:
    """Use instruction/input only for Alpaca; drop model answer tail."""
    if source != "alpaca":
        if source == "pubmedqa" and text.startswith("Question:"):
            q_end = text.find("\n\nAnswer:")
            if q_end != -1:
                return text[:q_end].strip()
        if source == "chatdoctor" and "Patient:" in text:
            doc = text.find("\n\nDoctor:")
            if doc != -1:
                return text[:doc].strip()
        return text

    parts = text.split("\n\n")
    if len(parts) <= 2:
        return text
    # Alpaca: instruction [+ input] + answer — drop final answer segment.
    return "\n\n".join(parts[:-1]).strip()


def keyword_is_medical(text: str) -> bool:
    lower = text.lower()
    if any(hint in lower for hint in NON_MEDICAL_HINTS):
        return False
    strong = sum(1 for kw in STRONG_MEDICAL_KEYWORDS if kw in lower)
    if strong >= 1:
        return True
    weak = sum(1 for kw in WEAK_MEDICAL_KEYWORDS if kw in lower)
    return weak >= 2


def format_chat(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    formatted = format_chat(tokenizer, prompt)
    inputs = tokenizer(
        formatted, return_tensors="pt", add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_yes_no(response: str) -> bool:
    first = response.strip().split()[0].lower() if response.strip() else ""
    return first.startswith("yes")


def parse_category(response: str) -> tuple[int, str]:
    """Map model output to (label, label_name). Defaults to General_Medicine."""
    cleaned = response.strip().split("\n")[0].strip()
    # Normalize: "General Medicine" -> "general_medicine"
    normalized = re.sub(r"[\s\-]+", "_", cleaned).lower()

    for name in CATEGORY_NAMES:
        key = name.lower()
        if key in normalized or normalized.startswith(key.replace("_", "")):
            return NAME_TO_LABEL[key], name

    # Partial match on distinctive substrings
    aliases = {
        "cardio": (0, "Cardiology"),
        "neuro": (1, "Neurology"),
        "oncol": (2, "Oncology"),
        "orthop": (3, "Orthopedics"),
        "general": (4, "General_Medicine"),
    }
    for prefix, (label, name) in aliases.items():
        if prefix in normalized:
            return label, name

    return 4, "General_Medicine"


def is_medical_sample(
    query_text: str, source: str, model, tokenizer,
) -> bool:
    if source in MEDICAL_SOURCES:
        return True
    if keyword_is_medical(query_text):
        return True
    prompt = MEDICAL_CHECK_PROMPT.format(text=truncate_text(query_text, 800))
    response = generate_response(model, tokenizer, prompt, max_new_tokens=8)
    return parse_yes_no(response)


def classify_sample(query_text: str, model, tokenizer) -> tuple[int, str]:
    prompt = ROUTING_PROMPT.format(text=truncate_text(query_text))
    response = generate_response(model, tokenizer, prompt, max_new_tokens=24)
    return parse_category(response)


def main():
    parser = argparse.ArgumentParser(description="Label medical routing categories")
    parser.add_argument(
        "--input",
        default="data/public_dataset/texts.jsonl",
        help="Input JSONL path",
    )
    parser.add_argument(
        "--output",
        default="data/routing_labels.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N samples (for testing)",
    )
    from src.platform import default_config_name

    parser.add_argument(
        "--config",
        default=default_config_name(),
        help="Config file path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sample_idx already present in output; append new labels",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model_id = config["models"]["qwen"]["model_id"]

    samples = load_samples(input_path)
    if args.dry_run is not None:
        samples = samples[: args.dry_run]
        log(f"Dry run: processing first {len(samples)} samples only")

    if args.resume:
        done_indices = load_labeled_indices(output_path)
        log(f"Resume: {len(done_indices)} samples already labeled")
    else:
        done_indices = set()
        output_path.write_text("")

    log(f"Loaded {len(samples)} samples from {input_path}")
    log(f"Loading Qwen model from {model_id}...")
    model, tokenizer = load_model(model_id)
    log("Model loaded.\n")

    labeled_count = len(done_indices)
    skipped_non_medical = 0

    try:
        with open(output_path, "a") as out_f:
            for idx, sample in enumerate(samples):
                if idx in done_indices:
                    continue

                text = sample["text"]
                source = sample.get("source", "unknown")
                query_text = extract_query_text(text, source)

                if not is_medical_sample(query_text, source, model, tokenizer):
                    skipped_non_medical += 1
                    if args.dry_run is not None:
                        log(f"  [{idx}] SKIP (non-medical) source={source}")
                    continue

                label, label_name = classify_sample(query_text, model, tokenizer)
                record = {
                    "sample_idx": idx,
                    "text": text,
                    "label": label,
                    "label_name": label_name,
                    "source": source,
                }
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                labeled_count += 1

                if args.dry_run is not None or labeled_count % 50 == 0:
                    preview = text[:80].replace("\n", " ")
                    log(
                        f"  [{idx}] {label_name} ({source}) "
                        f"| labeled={labeled_count} | {preview}..."
                    )

        records = []
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        counts = Counter(r["label_name"] for r in records)
        log("\n" + "=" * 60)
        log("  ROUTING LABEL SUMMARY")
        log("=" * 60)
        log(f"  Samples processed:     {len(samples)}")
        log(f"  Total labeled:       {labeled_count}")
        log(f"  Skipped (non-medical): {skipped_non_medical}")
        log("\n  Per category:")
        for _, name in CATEGORIES:
            log(f"    {name}: {counts.get(name, 0)}")
        log(f"\n  Output: {output_path}")
        log("=" * 60)

        if labeled_count < 500 and args.dry_run is None:
            log(
                f"\n  WARNING: Only {labeled_count} labeled samples "
                "(target is at least 500)."
            )
    finally:
        unload_model(model)


if __name__ == "__main__":
    main()
