#!/usr/bin/env python3
"""Train a 5-class medical routing head on specialist hidden states.

Reads specialist (Llama) hidden states from data/raw/{pair}/specialist/ and
routing labels from data/routing_labels.jsonl.  Trains a small linear
classifier (4096 -> 5) via L2-regularised softmax (multinomial logistic
regression solved with scipy L-BFGS).

Output: data/pairs/{pair}/routing_head.npz  containing weights (4096, 5)
and bias (5,).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import load_config

CATEGORIES = ["Cardiology", "Neurology", "Oncology", "Orthopedics", "General_Medicine"]
MEDICAL_SOURCES = {"pubmedqa", "chatdoctor"}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_labels(path: Path, medical_only: bool = True) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if medical_only and rec.get("source") not in MEDICAL_SOURCES:
                continue
            records.append(rec)
    return records


def load_specialist_states(
    raw_dir: Path, sample_indices: list[int], hidden_dim: int,
) -> np.ndarray:
    """Load and mean-pool specialist hidden states for given sample indices."""
    vectors = np.zeros((len(sample_indices), hidden_dim), dtype=np.float32)
    for i, idx in enumerate(sample_indices):
        path = raw_dir / f"{idx:05d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing specialist state: {path}")
        data = np.load(path)
        states = data["hidden_states"].astype(np.float32)
        vectors[i] = states.mean(axis=0)
        if (i + 1) % 200 == 0:
            log(f"  Loaded {i + 1}/{len(sample_indices)} states")
    return vectors


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def loss_and_grad(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lam: float,
) -> tuple[float, np.ndarray]:
    n, d = X.shape
    W = params[: d * n_classes].reshape(d, n_classes)
    b = params[d * n_classes:]

    logits = X @ W + b
    probs = softmax(logits)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), y] = 1.0

    loss = -np.sum(one_hot * np.log(probs + 1e-12)) / n + 0.5 * lam * np.sum(W ** 2)

    diff = (probs - one_hot) / n
    grad_W = X.T @ diff + lam * W
    grad_b = diff.sum(axis=0)

    grad = np.concatenate([grad_W.ravel(), grad_b])
    return loss, grad


def train(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int = 5,
    lam: float = 1e-3,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    params0 = np.zeros(d * n_classes + n_classes, dtype=np.float64)

    result = minimize(
        loss_and_grad,
        params0,
        args=(X.astype(np.float64), y, n_classes, lam),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "disp": True},
    )

    W = result.x[: d * n_classes].reshape(d, n_classes).astype(np.float32)
    b = result.x[d * n_classes:].astype(np.float32)
    return W, b


def evaluate(
    X: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray,
) -> dict:
    logits = X @ W + b
    preds = logits.argmax(axis=1)
    acc = (preds == y).mean()

    per_class = {}
    for i, name in enumerate(CATEGORIES):
        mask = y == i
        if mask.sum() == 0:
            per_class[name] = {"count": 0, "accuracy": 0.0}
            continue
        class_acc = (preds[mask] == i).mean()
        per_class[name] = {"count": int(mask.sum()), "accuracy": round(float(class_acc), 4)}

    return {"accuracy": round(float(acc), 4), "per_class": per_class}


def main():
    parser = argparse.ArgumentParser(description="Train routing head for HELIX")
    from src.platform import default_config_name

    parser.add_argument("--config", default=default_config_name())
    parser.add_argument("--labels", default="data/routing_labels.jsonl")
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--practitioner-id", default=None)
    parser.add_argument("--specialist-id", default=None)
    args = parser.parse_args()

    from src.model_catalog import resolve_pair_paths

    config = load_config(args.config)
    pr_id = args.practitioner_id or config.get("active_models", {}).get("practitioner", "qwen-2.5-7b")
    sp_id = args.specialist_id or config.get("active_models", {}).get("specialist", "llama-3.1-8b")
    pair_id = f"{pr_id}+{sp_id}"
    pair = resolve_pair_paths(config, pr_id, sp_id, Path("."))

    labels_path = Path(args.labels)
    raw_dir = Path(pair["raw_specialist_dir"])
    output_dir = Path(pair["pair_dir"])
    output_path = output_dir / "routing_head.npz"

    log(f"Pair: {pair_id}")
    log(f"Labels: {labels_path}")
    log(f"Specialist states: {raw_dir}")
    log(f"Output: {output_path}")

    labels = load_labels(labels_path, medical_only=True)
    log(f"\nLoaded {len(labels)} medical labels (filtered to pubmedqa + chatdoctor)")

    from collections import Counter
    dist = Counter(r["label_name"] for r in labels)
    for name in CATEGORIES:
        log(f"  {name}: {dist.get(name, 0)}")

    specialist_cfg = None
    for m in config.get("model_catalog", {}).get("specialists", []):
        if m["id"] == sp_id:
            specialist_cfg = m
            break
    hidden_dim = specialist_cfg["hidden_dim"] if specialist_cfg else 4096

    sample_indices = [r["sample_idx"] for r in labels]
    y_all = np.array([r["label"] for r in labels], dtype=np.int64)

    log(f"\nLoading {len(sample_indices)} specialist hidden states (mean-pooled)...")
    X_all = load_specialist_states(raw_dir, sample_indices, hidden_dim)
    log(f"X shape: {X_all.shape}, y shape: {y_all.shape}")

    n = len(X_all)
    n_val = max(1, int(n * args.val_split))
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_val, y_val = X_all[val_idx], y_all[val_idx]
    log(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    log(f"\nTraining (L-BFGS, lambda={args.lambda_reg}, max_iter={args.max_iter})...")
    W, b = train(X_train, y_train, n_classes=5, lam=args.lambda_reg, max_iter=args.max_iter)
    log(f"Weights: {W.shape}, Bias: {b.shape}")

    train_metrics = evaluate(X_train, y_train, W, b)
    val_metrics = evaluate(X_val, y_val, W, b)

    log(f"\nTrain accuracy: {train_metrics['accuracy']:.4f}")
    log(f"Val accuracy:   {val_metrics['accuracy']:.4f}")
    log("\nPer-class validation:")
    for name in CATEGORIES:
        info = val_metrics["per_class"][name]
        log(f"  {name}: {info['accuracy']:.4f} ({info['count']} samples)")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        weights=W,
        bias=b,
        categories=np.array(CATEGORIES),
        train_accuracy=train_metrics["accuracy"],
        val_accuracy=val_metrics["accuracy"],
        val_metrics=json.dumps(val_metrics),
        hidden_dim=hidden_dim,
        n_train=len(X_train),
        n_val=len(X_val),
    )
    log(f"\nSaved routing head to {output_path}")
    log(f"  weights: ({hidden_dim}, 5) float32")
    log(f"  bias: (5,) float32")


if __name__ == "__main__":
    main()
