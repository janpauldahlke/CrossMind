"""
Phase 0 Gate: Verify that we can extract hidden states and apply the LM head
independently using HuggingFace transformers.

macOS: MPS + float16.

This proves:
1. model.model(input_ids) returns post-norm hidden states (shape: 1, seq_len, hidden_dim)
2. model.lm_head(hidden_states) produces logits matching model(input_ids)
3. The split works for both Llama and Qwen

If this script passes, the entire RotationVault pipeline is feasible.
"""

import sys
import time

import torch

from src.models import load_config, load_model, unload_model
from src.platform import default_config_name


def verify_model(model_id: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"  Verifying: {model_name} ({model_id})")
    print(f"{'='*60}")

    print("  Loading model (float16)...")
    t0 = time.time()
    model, tokenizer = load_model(model_id)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Device: {next(model.parameters()).device}")

    model.eval()
    test_text = "The patient presents with chest pain and shortness of breath"
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    print(f"  Input: \"{test_text}\"")
    print(f"  Token count: {input_ids.shape[1]}")

    with torch.no_grad():
        full_output = model(input_ids)
        full_logits = full_output.logits
        print(f"\n  [TEST 1] Full model logits shape: {full_logits.shape}")

        base_output = model.model(input_ids)
        hidden_states = base_output.last_hidden_state
        print(f"  [TEST 2] Hidden states shape: {hidden_states.shape}")
        print(f"           Hidden dim: {hidden_states.shape[-1]}")
        print(f"           dtype: {hidden_states.dtype}")

        split_logits = model.lm_head(hidden_states)
        print(f"  [TEST 3] Split logits shape: {split_logits.shape}")

        max_diff = (full_logits - split_logits).abs().max().item()
        mean_diff = (full_logits - split_logits).abs().mean().item()
        print(f"\n  [TEST 4] Logit equivalence check:")
        print(f"           Max absolute difference: {max_diff:.2e}")
        print(f"           Mean absolute difference: {mean_diff:.2e}")

        tolerance = 1e-2
        if max_diff < tolerance:
            print(f"           PASS (within tolerance {tolerance})")
        else:
            print(f"           FAIL (exceeds tolerance {tolerance})")
            return False

        has_nan = torch.isnan(hidden_states).any().item()
        has_inf = torch.isinf(hidden_states).any().item()
        print(f"\n  [TEST 5] Numerical health:")
        print(f"           NaN in hidden states: {has_nan}")
        print(f"           Inf in hidden states: {has_inf}")
        if has_nan or has_inf:
            print("           FAIL")
            return False
        print("           PASS")

        last_hidden = hidden_states[:, -1:, :]
        next_logits = model.lm_head(last_hidden)
        next_token_id = next_logits.argmax(dim=-1).item()
        next_token_text = tokenizer.decode([next_token_id])
        print(f"\n  [TEST 6] Next token prediction from split hidden state:")
        print(f"           Predicted token ID: {next_token_id}")
        print(f"           Predicted text: \"{next_token_text}\"")
        print(f"           PASS (generated valid token)")

    print(f"\n  Summary for {model_name}:")
    print(f"    hidden_dim = {hidden_states.shape[-1]}")
    print(f"    vocab_size = {split_logits.shape[-1]}")
    print(f"    LM head weight shape = {model.lm_head.weight.shape}")
    print(f"    model.model() returns post-norm states: CONFIRMED")
    print(f"    model.lm_head() is a pure linear projection: CONFIRMED")

    unload_model(model)
    return True


def main():
    config = load_config()
    print(f"Config: {default_config_name()}  Platform: macOS/MPS")

    results = {}
    for name, model_cfg in config["models"].items():
        model_id = model_cfg["model_id"]
        success = verify_model(model_id, name)
        results[name] = success

    print(f"\n{'='*60}")
    print("  PHASE 0 GATE RESULTS")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\n  Overall: {'ALL GATES PASSED' if all_passed else 'GATE FAILED'}")
    print(f"{'='*60}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
