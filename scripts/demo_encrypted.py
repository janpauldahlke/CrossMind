#!/usr/bin/env python3
"""Phase 4 demo: Rotation encryption for cross-model generation.

Runs three scenarios on the same prompts to demonstrate that:
1. Correct key   encrypt + decrypt produces identical output to Phase 3.
2. No key        encrypted vectors fed to the LM head yield garbage.
3. Wrong key     decrypting with a different R produces garbage.

Generates a rotation key and saves it to data/rotation_key.npy if one does
not already exist.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from src.encryption import generate_rotation_key, save_key, load_key, verify_key
from src.generation import (
    extract_lm_head,
    generate_cross_model,
    load_generation_components,
    GenerationComponents,
)
from src.models import load_config

TEST_PROMPTS = [
    (
        "A 45-year-old male arrives at the ER with chest pain and"
        " shortness of breath. After initial examination, the doctor"
        " determines that the patient"
    ),
    (
        "A patient with Type 2 diabetes and chronic kidney disease"
        " should follow a treatment plan that includes"
    ),
    (
        "When a patient presents with sudden onset severe headache,"
        " the physician should first consider whether the cause is"
    ),
]

SCENARIOS = [
    ("Correct key (encrypt + decrypt)", "full"),
    ("No key (encrypted → LM head)", "no_decrypt"),
    ("Wrong key (decrypt with R_fake)", "wrong_key"),
]


def _ensure_lm_head(config: dict) -> None:
    """Extract Llama's LM head and tokenizer if not already on disk."""
    gen_cfg = config["generation"]
    lm_head_path = Path(gen_cfg["lm_head_path"])
    tokenizer_path = Path(gen_cfg["tokenizer_path"])

    if not lm_head_path.exists():
        print("=" * 60)
        print("  Extracting Llama LM head (one-time)")
        print("=" * 60)
        extract_lm_head(
            model_id=config["models"]["llama"]["model_id"],
            save_path=str(lm_head_path),
            tokenizer_save_path=str(tokenizer_path),
        )
        print()


def _ensure_rotation_key(config: dict) -> np.ndarray:
    """Generate and save a rotation key if one does not already exist."""
    enc_cfg = config["encryption"]
    key_path = Path(enc_cfg["key_path"])
    dim = enc_cfg["key_dim"]

    if key_path.exists():
        print(f"Loading existing rotation key from {key_path}")
        R = load_key(key_path)
    else:
        print(f"Generating {dim}×{dim} rotation key ...")
        t0 = time.perf_counter()
        R = generate_rotation_key(dim)
        elapsed = time.perf_counter() - t0
        print(f"  Generated in {elapsed:.3f}s")
        save_key(R, key_path)
        print(f"  Saved → {key_path}")

    stats = verify_key(R)
    print(
        f"  Orthogonality check: max_error={stats['max_error']:.2e}  "
        f"valid={stats['is_valid']}"
    )
    return R


def _truncate(text: str, max_chars: int = 120) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …"


def main() -> None:
    config = load_config()
    qwen_name = Path(config["models"]["qwen"]["model_id"]).name
    llama_name = Path(config["models"]["llama"]["model_id"]).name
    max_tokens = config["generation"].get("max_tokens", 50)

    # ------------------------------------------------------------------
    # Prerequisites
    # ------------------------------------------------------------------
    _ensure_lm_head(config)
    R = _ensure_rotation_key(config)

    # ------------------------------------------------------------------
    # Load components
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Loading generation components")
    print("=" * 60)
    t0 = time.time()
    components = load_generation_components(config)
    load_time = time.time() - t0
    print(f"Components loaded in {load_time:.1f}s")

    device = components.W_star.device
    R_tensor = torch.from_numpy(R).to(device)
    components.rotation_key = R_tensor

    R_fake = generate_rotation_key(config["encryption"]["key_dim"], seed=9999)
    R_fake_tensor = torch.from_numpy(R_fake).to(device)

    # ------------------------------------------------------------------
    # Run scenarios
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Phase 4: Encryption Demo  ({qwen_name} → {llama_name})")
    print(f"  max_tokens = {max_tokens}")
    print("=" * 60)

    for pi, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"PROMPT {pi}/{len(TEST_PROMPTS)}:")
        print(f"  {prompt}\n")

        for label, mode in SCENARIOS:
            wrong_key = R_fake_tensor if mode == "wrong_key" else None

            result = generate_cross_model(
                prompt,
                components,
                max_tokens=max_tokens,
                encryption_mode=mode,
                wrong_key=wrong_key,
            )

            coherent = mode == "full"
            tag = "COHERENT" if coherent else "GARBAGE "
            print(f"  [{tag}] {label}")
            print(f"           {_truncate(result.text)}")
            print(
                f"           tokens={result.num_tokens}  "
                f"{result.tokens_per_second:.1f} tok/s  "
                f"stop={result.stop_reason}"
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Phase 4 Complete: Rotation Encryption")
    print(sep)
    print(f"  Key dimensions:    {R.shape[0]}×{R.shape[1]}")
    print(f"  Key file:          {config['encryption']['key_path']}")
    print(f"  Prompts tested:    {len(TEST_PROMPTS)}")
    print(f"  Scenarios/prompt:  {len(SCENARIOS)}")
    print()
    print("  Expected results:")
    print("    ✓ Correct key  → output identical to Phase 3 (lossless rotation)")
    print("    ✗ No key       → unrecognizable tokens (encrypted vectors)")
    print("    ✗ Wrong key    → unrecognizable tokens (mismatched rotation)")


if __name__ == "__main__":
    main()
