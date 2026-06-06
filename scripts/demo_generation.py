#!/usr/bin/env python3
"""Phase 3 demo: Cross-model text generation.

Extracts Llama's LM head (if not already saved), loads all generation
components, runs 5 test prompts through the cross-model pipeline, and
saves results to data/generation_samples.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generation import (
    extract_lm_head,
    generate_cross_model,
    load_generation_components,
)
from src.models import load_config

TEST_PROMPTS = [
    (
        "A 45-year-old male presents with chest pain radiating to his left"
        " arm and diaphoresis. The specialist assessment indicates that"
    ),
    (
        "For a patient with Type 2 diabetes and chronic kidney disease,"
        " the recommended treatment approach involves"
    ),
    (
        "A 30-year-old woman with persistent fatigue, weight gain, and cold"
        " intolerance most likely has a condition affecting the"
    ),
    (
        "The primary mechanism of action of metformin in managing blood"
        " glucose levels involves"
    ),
    (
        "A post-operative patient on day 3 develops sudden shortness of"
        " breath and pleuritic chest pain. The most important immediate"
        " step is to"
    ),
]


def main():
    config = load_config()
    gen_cfg = config["generation"]
    llama_cfg = config["models"]["llama"]

    lm_head_path = Path(gen_cfg["lm_head_path"])
    tokenizer_path = Path(gen_cfg["tokenizer_path"])

    # ------------------------------------------------------------------
    # Extract LM head (one-time)
    # ------------------------------------------------------------------
    if not lm_head_path.exists():
        print("=" * 60)
        print("  Extracting Llama LM head (one-time)")
        print("=" * 60)
        extract_lm_head(
            model_id=llama_cfg["model_id"],
            save_path=str(lm_head_path),
            tokenizer_save_path=str(tokenizer_path),
        )
        print()

    # ------------------------------------------------------------------
    # Load components
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Loading generation components")
    print("=" * 60)
    t0 = time.time()
    components = load_generation_components(config)
    load_time = time.time() - t0
    print(f"Components loaded in {load_time:.1f}s\n")

    # ------------------------------------------------------------------
    # Run test prompts
    # ------------------------------------------------------------------
    max_tokens = gen_cfg.get("max_tokens", 50)
    results: list[dict] = []

    print("=" * 60)
    print(f"  Running {len(TEST_PROMPTS)} test prompts  (max_tokens={max_tokens})")
    print("=" * 60)

    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n--- Prompt {i}/{len(TEST_PROMPTS)} ---")
        qwen_name = Path(config["models"]["qwen"]["model_id"]).name
        llama_name = Path(config["models"]["llama"]["model_id"]).name
        print(f"INPUT  [{qwen_name}]:  {prompt}")

        result = generate_cross_model(prompt, components, max_tokens=max_tokens)

        print(f"OUTPUT [{llama_name}]: {result.text}")
        print(
            f"  tokens: {result.num_tokens}  |  "
            f"{result.tokens_per_second:.1f} tok/s  |  "
            f"stop: {result.stop_reason}"
        )

        results.append({
            "prompt": prompt,
            "generated": result.text,
            "num_tokens": result.num_tokens,
            "tokens_per_second": round(result.tokens_per_second, 2),
            "stop_reason": result.stop_reason,
        })

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = Path("data/generation_samples.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    speeds = [r["tokens_per_second"] for r in results]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0.0

    sep = "=" * 60
    print(f"\n{sep}")
    print("  Phase 3 Complete: Cross-Model Generation")
    print(sep)
    print(f"  Prompts run:       {len(results)}")
    print(f"  Avg tokens/sec:    {avg_speed:.1f}")
    print(f"  Results file:      {out_path}")
    for i, r in enumerate(results, 1):
        status = "EOS" if r["stop_reason"] == "eos" else f"{r['num_tokens']} tok"
        print(f"  Prompt {i}: {status}  ({r['tokens_per_second']:.1f} tok/s)")


if __name__ == "__main__":
    main()
