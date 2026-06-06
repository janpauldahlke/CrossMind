#!/usr/bin/env python3
"""Phase 2: Train linear alignment from Qwen's hidden space to Llama's.

Loads aligned chunk files produced by Phase 1, learns an affine mapping
(W*, b*) via ridge regression, evaluates on validation data, and persists
the result.

Train/val split handling:
    If data/aligned/train/ and data/aligned/val/ exist, they are used directly.
    Otherwise all chunks in data/aligned/ are used for BOTH training and
    evaluation.  This is because Phase 1's extract_states.py writes all chunks
    flat to data/aligned/ without preserving the dataset's split field.
    Validation metrics in this case are computed on training data and will be
    optimistically biased — acceptable during development.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.alignment import evaluate_alignment, learn_alignment


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train linear alignment map")
    parser.add_argument("--practitioner-id", default=None)
    parser.add_argument("--specialist-id", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger(__name__)

    from src.models import load_config

    config = load_config()

    if args.practitioner_id and args.specialist_id:
        from src.model_catalog import resolve_pair, get_model_entry

        from src.model_catalog import resolve_pair_paths

        pair = resolve_pair_paths(
            config, args.practitioner_id, args.specialist_id, Path("."),
        )
        practitioner = get_model_entry(config, "practitioner", args.practitioner_id)
        specialist = get_model_entry(config, "specialist", args.specialist_id)
        if not practitioner or not specialist:
            log.error("Unknown practitioner or specialist model id")
            sys.exit(1)

        aligned_dir = Path(pair["aligned_dir"])
        hidden_dim_a = specialist["hidden_dim"]
        hidden_dim_b = practitioner["hidden_dim"]
        map_path = Path(pair["map_path"])
        metrics_path = Path(pair["metrics_path"])
        log.info(
            "Training pair %s → %s", args.practitioner_id, args.specialist_id
        )
    else:
        aligned_dir = Path(config["extraction"]["aligned_dir"])
        hidden_dim_a = config["models"]["llama"]["hidden_dim"]
        hidden_dim_b = config["models"]["qwen"]["hidden_dim"]
        map_path = Path(config["alignment"]["map_path"])
        metrics_path = Path(config["alignment"]["metrics_path"])

    lambda_reg = config["alignment"]["lambda_reg"]

    # --- Resolve train / val directories ---
    train_dir = aligned_dir / "train"
    val_dir = aligned_dir / "val"

    if train_dir.exists() and any(train_dir.glob("chunk_*.npz")):
        log.info("Using pre-split directories: %s, %s", train_dir, val_dir)
    else:
        log.info(
            "No train/val subdirs in %s — using all chunks for training AND validation",
            aligned_dir,
        )
        train_dir = aligned_dir
        val_dir = aligned_dir

    # --- Train ---
    log.info("=" * 60)
    log.info("Learning alignment: dim %d → %d  (λ=%.1e)", hidden_dim_b, hidden_dim_a, lambda_reg)
    log.info("=" * 60)

    t0 = time.time()
    W_star, b_star, metadata = learn_alignment(
        train_dir, lambda_reg, hidden_dim_a, hidden_dim_b,
    )
    train_time = time.time() - t0

    log.info("Training complete in %.1fs", train_time)
    log.info("W* shape: %s  dtype: %s", W_star.shape, W_star.dtype)
    log.info("b* shape: %s  dtype: %s", b_star.shape, b_star.dtype)
    log.info("Condition number: %.2e", metadata["condition_number"])
    log.info("Training tokens: %d", metadata["n_train_tokens"])

    # --- Evaluate ---
    log.info("=" * 60)
    log.info("Evaluating on validation data: %s", val_dir)
    log.info("=" * 60)

    metrics = evaluate_alignment(W_star, b_star, val_dir, hidden_dim_a, hidden_dim_b)

    log.info(
        "Cosine similarity: %.4f ± %.4f  (median %.4f)",
        metrics["cosine_similarity_mean"],
        metrics["cosine_similarity_std"],
        metrics["cosine_similarity_median"],
    )
    log.info("MSE: %.6f", metrics["mse"])
    log.info("CKA: %.4f  (on %d tokens)", metrics["cka"], metrics["n_cka_subsample"])

    # --- Save alignment map ---
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        map_path,
        W_star=W_star,
        b_star=b_star,
        lambda_reg=np.float64(lambda_reg),
        n_train_tokens=np.int64(metadata["n_train_tokens"]),
    )
    log.info("Saved alignment map → %s", map_path)

    # --- Save metrics ---
    all_metrics = {
        **metrics,
        "train_time_seconds": round(train_time, 2),
        "lambda_reg": lambda_reg,
        "n_train_tokens": metadata["n_train_tokens"],
        "condition_number": metadata["condition_number"],
        "train_val_same_data": str(train_dir) == str(val_dir),
    }

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Saved metrics → %s", metrics_path)

    if args.practitioner_id and args.specialist_id:
        from src.generation import extract_lm_head

        lm_head_path = Path(pair["lm_head_path"])
        tokenizer_path = Path(pair["tokenizer_path"])
        if not lm_head_path.exists() or not tokenizer_path.is_dir():
            log.info("Extracting specialist LM head and tokenizer …")
            extract_lm_head(
                specialist["model_path"],
                str(lm_head_path),
                str(tokenizer_path),
            )

    # --- Summary ---
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Phase 2 Complete: Linear Alignment")
    print(sep)
    print(f"  W* shape:        {tuple(W_star.shape)}")
    print(f"  b* shape:        {tuple(b_star.shape)}")
    print(f"  λ:               {lambda_reg:.1e}")
    print(f"  Train tokens:    {metadata['n_train_tokens']:,}")
    print(f"  Condition #:     {metadata['condition_number']:.2e}")
    print(f"  Cosine sim:      {metrics['cosine_similarity_mean']:.4f}")
    print(f"  MSE:             {metrics['mse']:.6f}")
    print(f"  CKA:             {metrics['cka']:.4f}")
    print(f"  Train time:      {train_time:.1f}s")
    print(f"  Alignment map:   {map_path}")
    print(f"  Metrics file:    {metrics_path}")

    if metrics["cosine_similarity_mean"] < 0.5:
        log.warning(
            "Cosine similarity %.3f is below 0.5 target — see phase plan debugging checklist",
            metrics["cosine_similarity_mean"],
        )


if __name__ == "__main__":
    main()
