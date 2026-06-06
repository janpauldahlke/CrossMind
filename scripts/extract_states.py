"""
Phase 1.4: Batch hidden state extraction and token alignment pipeline.

Three-pass design (memory constrained -- one model loaded at a time):
  Pass 1: Load specialist, extract hidden states + offsets for all samples, save per-sample
  Pass 2: Unload specialist, load practitioner, extract hidden states + offsets, save per-sample
  Pass 3: Load saved offsets, compute token alignment, save aligned pairs in chunks

Outputs:
  data/raw/{pair}/specialist/{sample_id}.npz    -- hidden_states (seq, 4096) + offsets
  data/raw/{pair}/practitioner/{sample_id}.npz  -- hidden_states (seq, 3584) + offsets
  data/aligned/{pair}/chunk_{i}.npz             -- Z_A (n_pairs, 4096) + Z_B (n_pairs, 3584)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alignment import align_tokens, validate_alignment
from src.models import extract_hidden_states, load_model, unload_model


def load_config():
    from src.models import load_config as _load_config

    return _load_config()


def load_dataset(dataset_path: str, split: str = None) -> list[dict]:
    """Load samples from the prepared JSONL dataset."""
    samples = []
    with open(dataset_path) as f:
        for line in f:
            row = json.loads(line)
            if split is None or row.get("split") == split:
                samples.append(row)
    return samples


def extract_pass(
    model_id: str,
    model_name: str,
    samples: list[dict],
    output_dir: Path,
):
    """Run extraction for a single model over all samples."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check how many are already extracted (supports resume)
    existing = {p.stem for p in output_dir.glob("*.npz")}
    remaining = [
        (i, s) for i, s in enumerate(samples) if f"{i:05d}" not in existing
    ]

    if not remaining:
        print(f"  All {len(samples)} samples already extracted for {model_name}")
        return

    print(f"  Loading {model_name} ({model_id})...")
    model, tokenizer = load_model(model_id)
    print(f"  Model loaded. Extracting {len(remaining)} samples...")

    t0 = time.time()
    errors = 0

    for batch_idx, (i, sample) in enumerate(remaining):
        sample_id = f"{i:05d}"
        text = sample["text"]

        try:
            hidden, offsets = extract_hidden_states(model, tokenizer, text)

            offsets_arr = np.array(offsets, dtype=np.int32)
            np.savez_compressed(
                output_dir / f"{sample_id}.npz",
                hidden_states=hidden,
                offsets=offsets_arr,
            )
        except Exception as e:
            errors += 1
            print(f"    ERROR sample {sample_id}: {e}")
            continue

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (len(remaining) - batch_idx - 1) / rate
            print(
                f"    [{batch_idx + 1}/{len(remaining)}] "
                f"{rate:.1f} samples/s, ETA {eta / 60:.0f}min"
            )

    elapsed = time.time() - t0
    print(
        f"  Done: {len(remaining) - errors}/{len(remaining)} samples "
        f"in {elapsed / 60:.1f}min ({errors} errors)"
    )

    unload_model(model)


def alignment_pass(
    samples: list[dict],
    llama_dir: Path,
    qwen_dir: Path,
    aligned_dir: Path,
    chunk_size: int = 10000,
):
    """Load offsets from both models, align tokens, save aligned state pairs."""
    aligned_dir.mkdir(parents=True, exist_ok=True)

    print("  Aligning tokens and building state pair chunks...")

    chunk_za = []  # accumulates Llama hidden states (aligned)
    chunk_zb = []  # accumulates Qwen hidden states (aligned)
    chunk_idx = 0
    total_pairs = 0
    skipped = 0

    for i in range(len(samples)):
        sample_id = f"{i:05d}"
        llama_path = llama_dir / f"{sample_id}.npz"
        qwen_path = qwen_dir / f"{sample_id}.npz"

        if not llama_path.exists() or not qwen_path.exists():
            skipped += 1
            continue

        llama_data = np.load(llama_path)
        qwen_data = np.load(qwen_path)

        offsets_a = llama_data["offsets"].tolist()
        offsets_b = qwen_data["offsets"].tolist()

        # Convert flat [start, end, start, end...] back to tuples if needed
        offsets_a = [(offsets_a[j][0], offsets_a[j][1]) for j in range(len(offsets_a))]
        offsets_b = [(offsets_b[j][0], offsets_b[j][1]) for j in range(len(offsets_b))]

        pairs = align_tokens(offsets_a, offsets_b)

        if not pairs:
            skipped += 1
            continue

        hidden_a = llama_data["hidden_states"]  # (seq_a, 4096)
        hidden_b = qwen_data["hidden_states"]  # (seq_b, 3584)

        idx_a = [p[0] for p in pairs]
        idx_b = [p[1] for p in pairs]

        # Select aligned rows
        za = hidden_a[idx_a]  # (n_pairs, 4096)
        zb = hidden_b[idx_b]  # (n_pairs, 3584)

        chunk_za.append(za)
        chunk_zb.append(zb)
        total_pairs += len(pairs)

        # Flush chunk if large enough
        if total_pairs >= (chunk_idx + 1) * chunk_size:
            _save_chunk(chunk_za, chunk_zb, aligned_dir, chunk_idx)
            chunk_za = []
            chunk_zb = []
            chunk_idx += 1

        if (i + 1) % 500 == 0:
            print(f"    [{i + 1}/{len(samples)}] {total_pairs} pairs so far")

    # Save remaining
    if chunk_za:
        _save_chunk(chunk_za, chunk_zb, aligned_dir, chunk_idx)
        chunk_idx += 1

    print(
        f"  Done: {total_pairs} aligned pairs in {chunk_idx} chunks "
        f"({skipped} samples skipped)"
    )


def _save_chunk(za_list, zb_list, aligned_dir, chunk_idx):
    za = np.concatenate(za_list, axis=0)
    zb = np.concatenate(zb_list, axis=0)
    path = aligned_dir / f"chunk_{chunk_idx:03d}.npz"
    np.savez_compressed(path, Z_A=za, Z_B=zb)
    print(f"    Saved {path.name}: Z_A {za.shape}, Z_B {zb.shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract hidden states for alignment")
    parser.add_argument("--practitioner-id", default=None)
    parser.add_argument("--specialist-id", default=None)
    args = parser.parse_args()

    config = load_config()
    dataset_path = config["extraction"]["dataset_path"]
    chunk_size = config["alignment"]["chunk_size"]

    if args.practitioner_id and args.specialist_id:
        from src.model_catalog import resolve_pair, get_model_entry

        pair = resolve_pair(config, args.practitioner_id, args.specialist_id)

        practitioner = get_model_entry(config, "practitioner", args.practitioner_id)
        specialist = get_model_entry(config, "specialist", args.specialist_id)
        if not practitioner or not specialist:
            print("ERROR: Unknown practitioner or specialist model id")
            sys.exit(1)

        raw_practitioner = Path(pair["raw_practitioner_dir"])
        raw_specialist = Path(pair["raw_specialist_dir"])
        aligned_dir = Path(pair["aligned_dir"])
        practitioner_id = args.practitioner_id
        specialist_id = args.specialist_id
    else:
        practitioner = {"model_path": config["models"]["qwen"]["model_id"]}
        specialist = {"model_path": config["models"]["llama"]["model_id"]}
        raw_dir = Path(config["extraction"]["raw_dir"])
        raw_practitioner = raw_dir / "qwen"
        raw_specialist = raw_dir / "llama"
        aligned_dir = Path(config["extraction"]["aligned_dir"])
        practitioner_id = "qwen"
        specialist_id = "llama"

    if not Path(dataset_path).exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Run `python scripts/prepare_data.py` first.")
        sys.exit(1)

    samples = load_dataset(dataset_path)
    print(f"Loaded {len(samples)} samples from {dataset_path}")
    print(f"Pair: {practitioner_id} → {specialist_id}")

    print(f"\n{'='*60}")
    print("  Pass 1: Specialist hidden state extraction")
    print(f"{'='*60}")
    extract_pass(
        specialist["model_path"], "specialist", samples, raw_specialist,
    )

    print(f"\n{'='*60}")
    print("  Pass 2: Practitioner hidden state extraction")
    print(f"{'='*60}")
    extract_pass(
        practitioner["model_path"], "practitioner", samples, raw_practitioner,
    )

    print(f"\n{'='*60}")
    print("  Pass 3: Token alignment")
    print(f"{'='*60}")
    alignment_pass(
        samples,
        raw_specialist,
        raw_practitioner,
        aligned_dir,
        chunk_size=chunk_size,
    )

    print(f"\n{'='*60}")
    print("  Extraction complete!")
    print(f"{'='*60}")
    specialist_count = len(list(raw_specialist.glob("*.npz")))
    practitioner_count = len(list(raw_practitioner.glob("*.npz")))
    chunk_count = len(list(aligned_dir.glob("chunk_*.npz")))
    print(f"  Specialist samples:   {specialist_count}")
    print(f"  Practitioner samples: {practitioner_count}")
    print(f"  Aligned chunks:       {chunk_count}")

    if chunk_count > 0:
        c = np.load(aligned_dir / "chunk_000.npz")
        print(f"  First chunk: Z_A {c['Z_A'].shape}, Z_B {c['Z_B'].shape}")
        if args.practitioner_id and args.specialist_id:
            spec_dim = specialist.get("hidden_dim", c["Z_A"].shape[1])
            prac_dim = practitioner.get("hidden_dim", c["Z_B"].shape[1])
        else:
            spec_dim = config["models"]["llama"]["hidden_dim"]
            prac_dim = config["models"]["qwen"]["hidden_dim"]
        assert c["Z_A"].shape[1] == spec_dim
        assert c["Z_B"].shape[1] == prac_dim
        print("  Dimension check: PASS")


if __name__ == "__main__":
    main()
