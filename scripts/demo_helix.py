#!/usr/bin/env python3
"""HELIX demo — encrypted medical routing via CKKS.

For each test prompt:
  1. Qwen encodes the query (Party B)
  2. Apply alignment W* to get h_aligned (Party B)
  3. Encrypt h_aligned with CKKS (Party B — only B has the secret key)
  4. Homomorphic matmul: ct @ routing_head + bias (Party A — no decryption)
  5. Party B decrypts result → department label
  6. Compare with plaintext classification (must match)
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.helix import (
    CATEGORIES,
    create_context,
    load_routing_head,
    run_helix_classification,
)
from src.models import load_config, load_model, unload_model

warnings.filterwarnings("ignore", category=FutureWarning)

DEMO_PROMPTS = [
    "Patient presents with acute chest pain radiating to the left arm, elevated troponin, and ST-segment elevation on ECG.",
    "A 65-year-old with sudden onset severe headache, neck stiffness, photophobia, and altered consciousness.",
    "Biopsy confirms invasive ductal carcinoma, ER-positive, HER2-negative, with lymph node involvement.",
    "Elderly patient with displaced femoral neck fracture after a fall, unable to bear weight.",
    "30-year-old female with fatigue, weight gain, cold intolerance, and elevated TSH levels.",
]

EXPECTED = ["Cardiology", "Neurology", "Oncology", "Orthopedics", "General_Medicine"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description="HELIX encrypted routing demo")
    from src.platform import default_config_name

    parser.add_argument("--config", default=default_config_name())
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()

    from src.model_catalog import resolve_pair_paths

    config = load_config(args.config)

    pr_id = config.get("active_models", {}).get("practitioner", "qwen-2.5-7b")
    sp_id = config.get("active_models", {}).get("specialist", "llama-3.1-8b")
    pair_id = f"{pr_id}+{sp_id}"
    pair = resolve_pair_paths(config, pr_id, sp_id, Path("."))

    map_path = Path(pair["map_path"])
    head_path = Path(pair["pair_dir"]) / "routing_head.npz"

    log("=" * 70)
    log("  HELIX — Homomorphic Encrypted Medical Routing Demo")
    log("=" * 70)
    log(f"  Pair: {pair_id}")
    log(f"  Alignment map: {map_path}")
    log(f"  Routing head: {head_path}")

    alignment = np.load(map_path)
    W_star = alignment["W_star"]
    b_star = alignment["b_star"]
    log(f"  W*: {W_star.shape}, b*: {b_star.shape}")

    weights, bias = load_routing_head(head_path)
    log(f"  Routing head: {weights.shape}, bias: {bias.shape}")

    model_id = config["models"]["qwen"]["model_id"]
    log(f"\n  Loading practitioner model: {model_id}")
    qwen_cfg = config["models"]["qwen"]
    model, tokenizer = load_model(model_id)
    log("  Model loaded.")

    log("\n  Creating CKKS context (one-time key generation)...")
    t0 = time.perf_counter()
    ctx = create_context()
    log(f"  Context ready in {(time.perf_counter() - t0) * 1000:.0f} ms")

    prompts = DEMO_PROMPTS
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]

    results = []
    log("\n" + "=" * 70)

    for i, prompt in enumerate(prompts):
        log(f"\n  Prompt {i + 1}: {prompt[:80]}...")

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(model.device)

        with torch.no_grad():
            output = model.model(input_ids)
            h_B = output.last_hidden_state[0, -1, :].cpu().numpy().astype(np.float32)

        h_aligned = h_B @ W_star + b_star

        result = run_helix_classification(h_aligned, weights, bias, ctx=ctx)
        results.append(result)

        expected = EXPECTED[i] if i < len(EXPECTED) else "?"
        match_str = "MATCH" if result.label_name == expected else "MISMATCH"
        crypto_str = "OK" if result.plaintext_matches else "DIFFERS"

        log(f"  → Routed to: {result.label_name} (expected: {expected}) [{match_str}]")
        log(f"    Encrypted == Plaintext: {crypto_str}")
        log(f"    Confidences: {dict(zip(CATEGORIES, result.confidences))}")
        log(f"    Timing: encrypt={result.encrypt_ms}ms  compute={result.compute_ms}ms  "
            f"decrypt={result.decrypt_ms}ms  total={result.total_ms}ms")

    unload_model(model)

    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)

    correct = sum(1 for r, e in zip(results, EXPECTED) if r.label_name == e)
    crypto_ok = sum(1 for r in results if r.plaintext_matches)
    avg_total = np.mean([r.total_ms for r in results])
    avg_encrypt = np.mean([r.encrypt_ms for r in results])
    avg_compute = np.mean([r.compute_ms for r in results])
    avg_decrypt = np.mean([r.decrypt_ms for r in results])

    log(f"  Routing accuracy:        {correct}/{len(results)}")
    log(f"  Crypto correctness:      {crypto_ok}/{len(results)} (encrypted == plaintext)")
    log(f"  Avg total latency:       {avg_total:.0f} ms")
    log(f"  Avg encrypt:             {avg_encrypt:.0f} ms")
    log(f"  Avg homomorphic compute: {avg_compute:.0f} ms")
    log(f"  Avg decrypt:             {avg_decrypt:.0f} ms")
    log()
    log("  Party A (server) never held the secret key or saw plaintext vectors.")
    log("  The routing decision was computed entirely on ciphertext.")
    log("=" * 70)


if __name__ == "__main__":
    main()
