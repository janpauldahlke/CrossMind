#!/usr/bin/env python3
"""Phase 5 demo: End-to-end two-party encrypted cross-model generation.

Starts the Party A server as a subprocess, waits for it to become healthy,
then runs the Party B client against it with three medical prompts.

Usage:
    python scripts/demo_e2e.py
    python scripts/demo_e2e.py --max-tokens 50
"""

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import load_config


def _wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll the server's /health endpoint until it responds."""
    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
            if resp.status_code == 200:
                return True
        except httpx.ConnectError:
            pass
        time.sleep(0.5)
    return False


def main():
    config = load_config()
    net = config["network"]
    host = net["party_a"].get("host", "0.0.0.0")
    port = net["party_a"].get("port", 8420)
    server_url = net["party_b"].get("server_url", f"http://localhost:{port}")
    max_tokens_default = config["generation"].get("max_tokens", 100)

    qwen_name = Path(config["models"]["qwen"]["model_id"]).name
    llama_name = Path(config["models"]["llama"]["model_id"]).name

    parser = argparse.ArgumentParser(description="CrossMind end-to-end demo")
    parser.add_argument("--max-tokens", type=int, default=max_tokens_default)
    parser.add_argument("--passphrase", default="hackathon2026")
    parser.add_argument("--use-handshake", action="store_true")
    args = parser.parse_args()

    sep = "=" * 60

    print(sep)
    print("  Phase 5: Two-Party Encrypted Cross-Model Generation")
    print(f"  {qwen_name} (Party B)  →  {llama_name} (Party A)")
    print(sep)

    # ------------------------------------------------------------------
    # Start server
    # ------------------------------------------------------------------
    print(f"\n[demo] Starting server on {host}:{port} ...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.server:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "warning",
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        print(f"[demo] Waiting for server at {server_url} ...")
        if not _wait_for_server(server_url, timeout=60.0):
            print("[demo] ERROR: Server did not start in time.", file=sys.stderr)
            server_proc.terminate()
            sys.exit(1)
        print("[demo] Server is ready.\n")

        # ------------------------------------------------------------------
        # Run client
        # ------------------------------------------------------------------
        from src.client import CrossMindClient

        from src.model_catalog import build_runtime_config, get_active_ids

        pr_id, sp_id = get_active_ids(config)
        config = build_runtime_config(config, pr_id, sp_id)

        client = CrossMindClient(config, server_url=server_url)
        if args.use_handshake:
            client.handshake()
        else:
            client.set_passphrase(args.passphrase)
            client.sync_passphrase_to_server(args.passphrase)
            print(f"[demo] Passphrase key exchange OK")

        prompts = [
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

        print(f"\n{sep}")
        print(f"  Generating with {len(prompts)} prompts  (max_tokens={args.max_tokens})")
        print(sep)

        all_results = []

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'─' * 60}")
            print(f"PROMPT {i}/{len(prompts)}:")
            print(f"  {prompt}\n")
            print("OUTPUT: ", end="")

            result = client.generate(prompt, max_tokens=args.max_tokens)
            all_results.append(result)

            print(
                f"\n  tokens={result['num_tokens']}  "
                f"{result['tokens_per_second']:.1f} tok/s  "
                f"avg_step={result['avg_step_ms']:.0f}ms  "
                f"total={result['elapsed_seconds']:.1f}s  "
                f"stop={result['stop_reason']}"
            )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        total_tokens = sum(r["num_tokens"] for r in all_results)
        total_time = sum(r["elapsed_seconds"] for r in all_results)
        avg_step = (
            sum(r["avg_step_ms"] for r in all_results) / len(all_results)
            if all_results else 0.0
        )
        overall_tps = total_tokens / total_time if total_time > 0 else 0.0

        print(f"\n{sep}")
        print("  Phase 5 Complete: Two-Party Network")
        print(sep)
        print(f"  Party B (encoder): {qwen_name}")
        print(f"  Party A (decoder): {llama_name}")
        print(f"  Server address:    {server_url}")
        print(f"  Prompts run:       {len(prompts)}")
        print(f"  Total tokens:      {total_tokens}")
        print(f"  Overall speed:     {overall_tps:.1f} tok/s")
        print(f"  Avg step latency:  {avg_step:.0f}ms  (compute + network)")
        print()
        for i, r in enumerate(all_results, 1):
            status = "EOS" if r["stop_reason"] == "eos" else f"{r['num_tokens']} tok"
            print(
                f"  Prompt {i}: {status}  "
                f"({r['tokens_per_second']:.1f} tok/s, "
                f"avg {r['avg_step_ms']:.0f}ms/step)"
            )
        print()

    finally:
        print("[demo] Shutting down server ...")
        server_proc.send_signal(signal.SIGINT)
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("[demo] Done.")


if __name__ == "__main__":
    main()
