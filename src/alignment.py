"""
Token alignment between different tokenizers.

Given two tokenizers that segment the same text differently, produces
(index_a, index_b) pairs that align tokens at corresponding character
positions. This is necessary because Llama and Qwen use different
tokenization schemes.

Approach follows the paper: for each token in model A, find the earliest
token in model B whose character end position >= model A's token end position.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.linalg import solve

logger = logging.getLogger(__name__)


def align_tokens(
    offsets_a: list[tuple[int, int]],
    offsets_b: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Align tokens between two tokenizers based on character offsets.

    For each token in model A, find the token in model B whose character
    span end is >= model A's token span end (earliest such match).

    Args:
        offsets_a: (char_start, char_end) per token from tokenizer A
        offsets_b: (char_start, char_end) per token from tokenizer B

    Returns:
        List of (index_a, index_b) aligned pairs.
    """
    if not offsets_a or not offsets_b:
        return []

    pairs = []
    j = 0
    for i, (_, end_a) in enumerate(offsets_a):
        # Advance j until B's token end >= A's token end
        while j < len(offsets_b) and offsets_b[j][1] < end_a:
            j += 1
        if j >= len(offsets_b):
            break
        pairs.append((i, j))

    return pairs


def validate_alignment(
    offsets_a: list[tuple[int, int]],
    offsets_b: list[tuple[int, int]],
    pairs: list[tuple[int, int]],
    text_len: int,
) -> dict:
    """Check alignment quality.

    Returns a dict with coverage stats and any issues found.
    """
    if not pairs:
        return {
            "valid": False,
            "reason": "no pairs produced",
            "coverage_a": 0.0,
            "coverage_b": 0.0,
        }

    indices_a = {p[0] for p in pairs}
    indices_b = {p[1] for p in pairs}

    coverage_a = len(indices_a) / len(offsets_a) if offsets_a else 0.0
    coverage_b = len(indices_b) / len(offsets_b) if offsets_b else 0.0

    # Check that aligned tokens cover the full text span
    chars_covered_a = set()
    for idx in indices_a:
        s, e = offsets_a[idx]
        chars_covered_a.update(range(s, e))

    chars_covered_b = set()
    for idx in indices_b:
        s, e = offsets_b[idx]
        chars_covered_b.update(range(s, e))

    # Monotonicity: indices should be non-decreasing in both dimensions
    monotonic = all(
        pairs[k][0] <= pairs[k + 1][0] and pairs[k][1] <= pairs[k + 1][1]
        for k in range(len(pairs) - 1)
    )

    return {
        "valid": monotonic and coverage_a > 0.5,
        "num_pairs": len(pairs),
        "coverage_a": coverage_a,
        "coverage_b": coverage_b,
        "char_coverage_a": len(chars_covered_a) / text_len if text_len > 0 else 0.0,
        "char_coverage_b": len(chars_covered_b) / text_len if text_len > 0 else 0.0,
        "monotonic": monotonic,
    }


# ---------------------------------------------------------------------------
# Phase 2: Linear alignment (ridge regression)
# ---------------------------------------------------------------------------


def learn_alignment(
    aligned_dir: str | Path,
    lambda_reg: float,
    hidden_dim_a: int,
    hidden_dim_b: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Learn affine mapping W*, b* from model B's hidden space to model A's.

    Uses incremental sufficient statistics so the full Z_A / Z_B matrices
    never need to reside in memory simultaneously.

    Precision pipeline:
        fp16 (disk) → fp32 (matmul) → fp64 (accumulation & solve) → fp32 (output)

    Args:
        aligned_dir: Directory containing chunk_*.npz files (each with Z_A, Z_B).
        lambda_reg: L2 regularisation coefficient.
        hidden_dim_a: Dimensionality of model A (target, e.g. 4096 for Llama).
        hidden_dim_b: Dimensionality of model B (source, e.g. 3584 for Qwen).

    Returns:
        W_star: (hidden_dim_b, hidden_dim_a) float32 mapping matrix.
        b_star: (hidden_dim_a,) float32 bias vector.
        metadata: Training statistics dict.
    """
    aligned_dir = Path(aligned_dir)
    chunk_files = sorted(aligned_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files in {aligned_dir}")

    ZtZ = np.zeros((hidden_dim_b, hidden_dim_b), dtype=np.float64)
    ZtA = np.zeros((hidden_dim_b, hidden_dim_a), dtype=np.float64)
    sum_A = np.zeros(hidden_dim_a, dtype=np.float64)
    sum_B = np.zeros(hidden_dim_b, dtype=np.float64)
    n_total = 0

    for path in chunk_files:
        chunk = np.load(path)
        Z_A = chunk["Z_A"].astype(np.float32)
        Z_B = chunk["Z_B"].astype(np.float32)
        assert Z_A.shape == (Z_B.shape[0], hidden_dim_a), (
            f"Z_A shape {Z_A.shape} vs expected (*, {hidden_dim_a})"
        )
        assert Z_B.shape[1] == hidden_dim_b, (
            f"Z_B dim {Z_B.shape[1]} != {hidden_dim_b}"
        )

        # fp32 matmul results are upcast to fp64 by the in-place +=
        ZtZ += Z_B.T @ Z_B
        ZtA += Z_B.T @ Z_A
        sum_A += Z_A.sum(axis=0)
        sum_B += Z_B.sum(axis=0)
        n_total += Z_B.shape[0]
        logger.info("Loaded %s: %d tokens (cumulative: %d)", path.name, Z_B.shape[0], n_total)

    mean_A = sum_A / n_total
    mean_B = sum_B / n_total

    # Mean-center the sufficient statistics
    ZtZ -= n_total * np.outer(mean_B, mean_B)
    ZtA -= n_total * np.outer(mean_B, mean_A)

    lhs = ZtZ + lambda_reg * np.eye(hidden_dim_b, dtype=np.float64)

    cond = float(np.linalg.cond(lhs))
    logger.info("Condition number of (ZtZ + λI): %.2e", cond)
    if cond > 1e8:
        logger.warning(
            "Condition number %.2e exceeds 1e8 — consider increasing lambda_reg", cond
        )

    W_star = solve(lhs, ZtA, assume_a="pos")
    b_star = mean_A - mean_B @ W_star

    if np.any(np.isnan(W_star)) or np.any(np.isinf(W_star)):
        raise ValueError("W_star contains NaN or Inf — numerical instability")

    W_star = W_star.astype(np.float32)
    b_star = b_star.astype(np.float32)

    metadata = {
        "n_chunks": len(chunk_files),
        "n_train_tokens": int(n_total),
        "lambda_reg": lambda_reg,
        "condition_number": cond,
        "W_shape": list(W_star.shape),
        "b_shape": list(b_star.shape),
    }
    return W_star, b_star, metadata


def evaluate_alignment(
    W_star: np.ndarray,
    b_star: np.ndarray,
    val_dir: str | Path,
    hidden_dim_a: int,
    hidden_dim_b: int,
) -> dict:
    """Evaluate alignment quality on validation chunks.

    Metrics computed:
        - Mean / std / median cosine similarity (per-token)
        - Reconstruction MSE (||pred − actual||² / N_tokens)
        - Linear CKA on a random 5000-token subsample

    Processes chunks one at a time to limit peak memory.
    """
    val_dir = Path(val_dir)
    chunk_files = sorted(val_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.npz files in {val_dir}")

    all_cosines: list[np.ndarray] = []
    mse_sum = 0.0
    n_total = 0

    for path in chunk_files:
        chunk = np.load(path)
        Z_A = chunk["Z_A"].astype(np.float32)
        Z_B = chunk["Z_B"].astype(np.float32)

        pred = Z_B @ W_star + b_star
        dot = np.sum(pred * Z_A, axis=1)
        cos = dot / (np.linalg.norm(pred, axis=1) * np.linalg.norm(Z_A, axis=1) + 1e-8)
        all_cosines.append(cos)

        mse_sum += float(np.sum((pred - Z_A) ** 2))
        n_total += Z_A.shape[0]

    cosines = np.concatenate(all_cosines)
    mse = mse_sum / n_total

    # CKA on a deterministic random subsample (second pass over chunks)
    cka_n = min(5000, n_total)
    rng = np.random.RandomState(42)
    sample_idx = np.sort(rng.choice(n_total, size=cka_n, replace=False))

    cka_pred_parts: list[np.ndarray] = []
    cka_actual_parts: list[np.ndarray] = []
    offset = 0

    for path in chunk_files:
        chunk = np.load(path)
        Z_A = chunk["Z_A"].astype(np.float32)
        Z_B = chunk["Z_B"].astype(np.float32)
        chunk_len = Z_A.shape[0]

        mask = (sample_idx >= offset) & (sample_idx < offset + chunk_len)
        local = sample_idx[mask] - offset
        if len(local) > 0:
            cka_pred_parts.append(Z_B[local] @ W_star + b_star)
            cka_actual_parts.append(Z_A[local])

        offset += chunk_len

    cka = _linear_cka(
        np.concatenate(cka_pred_parts),
        np.concatenate(cka_actual_parts),
    )

    return {
        "cosine_similarity_mean": float(np.mean(cosines)),
        "cosine_similarity_std": float(np.std(cosines)),
        "cosine_similarity_median": float(np.median(cosines)),
        "mse": mse,
        "cka": cka,
        "n_val_tokens": int(n_total),
        "n_cka_subsample": int(cka_n),
    }


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment.

    CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F · ||Yc^T Yc||_F)
    where Xc, Yc are column-centred.
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = float(np.sum(YtX**2))
    hsic_xx = float(np.sum(XtX**2))
    hsic_yy = float(np.sum(YtY**2))

    return hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
