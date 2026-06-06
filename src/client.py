"""
Party B client: Qwen-based medical query encoder with encrypted transport.

Loads the Qwen model (float16), the learned alignment map (W*, b*),
and a rotation key R.  For each autoregressive step the client:

    1. Runs Qwen's inner model forward pass (with KV cache).
    2. Transforms the last hidden state through the alignment map.
    3. Encrypts the aligned vector with the rotation key.
    4. Sends the encrypted vector to Party A's /infer endpoint.
    5. Receives the decoded token and feeds it back for the next step.

The wire protocol uses base64-encoded float32 byte buffers.
"""

import argparse
import base64
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch
from transformers import AutoTokenizer

from src.encryption import (
    derive_rotation_key,
    generate_rotation_key,
    key_fingerprint,
    load_key,
    save_key,
    verify_key,
)
from src.generation import _encode_continuation
from src.models import load_config, load_model
from src.two_party import TwoPartyContext, generate_two_party_stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_vector(v: np.ndarray) -> str:
    """float32 numpy vector → base64 string."""
    return base64.b64encode(v.astype(np.float32).tobytes()).decode()


def _encode_matrix(m: np.ndarray) -> str:
    """float32 numpy matrix → base64 string."""
    return base64.b64encode(m.astype(np.float32).tobytes()).decode()


def _get_stop_token_ids(tokenizer) -> set[int]:
    """Collect stop / end-of-turn token IDs from the specialist tokenizer."""
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    unk_id = tokenizer.unk_token_id
    for name in [
        "<|eot_id|>", "<|end_of_text|>",  # Llama 3
        "</s>",                            # Mistral / SentencePiece
        "<|im_end|>",                      # Qwen / ChatML
        "<|endoftext|>",                   # GPT-NeoX family
    ]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and (unk_id is None or tid != unk_id):
            stop_ids.add(tid)
    return stop_ids


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class CrossMindClient:
    """Party B: encodes queries with Qwen and sends encrypted vectors."""

    def __init__(self, config: dict, server_url: str | None = None):
        self.config = config
        self.server_url = server_url or config["network"]["party_b"]["server_url"]

        qwen_cfg = config["models"]["qwen"]
        map_path = config["alignment"]["map_path"]
        enc_cfg = config["encryption"]

        # Qwen model
        print("[client] Loading Qwen model ...")
        self.qwen_model, self.qwen_tokenizer = load_model(qwen_cfg["model_id"])
        self.device = next(self.qwen_model.parameters()).device

        # Alignment map
        print(f"[client] Loading alignment map from {map_path} ...")
        alignment = np.load(map_path)
        self.W_star = torch.from_numpy(alignment["W_star"].astype(np.float32)).to(self.device)
        self.b_star = torch.from_numpy(alignment["b_star"].astype(np.float32)).to(self.device)
        print(f"[client]   W* {tuple(self.W_star.shape)}  b* {tuple(self.b_star.shape)}")

        # Llama tokenizer (for stop-token detection)
        tok_path = config["generation"]["tokenizer_path"]
        print(f"[client] Loading Llama tokenizer from {tok_path} ...")
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            tok_path, clean_up_tokenization_spaces=False,
        )
        self.stop_ids = _get_stop_token_ids(self.llama_tokenizer)

        # Rotation key
        key_path = Path(enc_cfg["key_path"])
        key_dim = enc_cfg["key_dim"]
        if key_path.exists():
            print(f"[client] Loading rotation key from {key_path} ...")
            self.R = load_key(str(key_path))
        else:
            print(f"[client] Generating {key_dim}×{key_dim} rotation key ...")
            self.R = generate_rotation_key(key_dim)
            save_key(self.R, str(key_path))
            print(f"[client]   Saved → {key_path}")

        stats = verify_key(self.R)
        print(f"[client]   Orthogonality: max_err={stats['max_error']:.2e}  valid={stats['is_valid']}")

        self.R_torch = torch.from_numpy(self.R).to(self.device)
        self.hidden_dim = key_dim
        self.http = httpx.Client(timeout=30.0)
        self._lm_head = None
        lm_head_path = config["generation"].get("lm_head_path")
        if lm_head_path and Path(lm_head_path).exists():
            self._lm_head = np.load(lm_head_path).astype(np.float32)

    # ------------------------------------------------------------------
    # Key setup
    # ------------------------------------------------------------------

    def set_passphrase(self, passphrase: str) -> dict:
        """Derive rotation key from shared passphrase (no matrix on wire)."""
        self.R = derive_rotation_key(self.hidden_dim, passphrase)
        stats = verify_key(self.R)
        self.R_torch = torch.from_numpy(self.R).to(self.device)
        fp = key_fingerprint(self.R)
        return {"fingerprint": fp, "orthogonality_ok": stats["is_valid"]}

    def sync_passphrase_to_server(self, passphrase: str) -> dict:
        """Ask specialist server to derive the same key locally."""
        resp = self.http.post(
            f"{self.server_url}/api/session/key",
            json={"passphrase": passphrase},
        )
        resp.raise_for_status()
        return resp.json()

    def handshake(self) -> dict:
        """Send the rotation key to Party A."""
        print(f"[client] Sending rotation key to {self.server_url}/handshake ...")
        resp = self.http.post(
            f"{self.server_url}/handshake",
            json={
                "key_b64": _encode_matrix(self.R),
                "dim": self.R.shape[0],
            },
        )
        resp.raise_for_status()
        result = resp.json()
        print(f"[client] Handshake OK: {result}")
        return result

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    SYSTEM_PROMPT = (
        "You are a medical specialist providing brief clinical assessments. "
        "Give a concise diagnosis or recommendation in 1-2 sentences. "
        "Do not use multiple choice format. Do not list options. "
        "Respond directly with your clinical assessment."
    )

    def _two_party_ctx(self) -> TwoPartyContext:
        return TwoPartyContext(
            qwen_model=self.qwen_model,
            qwen_tokenizer=self.qwen_tokenizer,
            llama_tokenizer=self.llama_tokenizer,
            W_star=self.W_star,
            b_star=self.b_star,
            R_torch=self.R_torch,
            stop_ids=self.stop_ids,
            device=self.device,
            server_url=self.server_url,
            http=self.http,
            hidden_dim=self.hidden_dim,
            lm_head=self._lm_head,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        print_tokens: bool = True,
        system_prompt: str | None = SYSTEM_PROMPT,
    ) -> dict:
        """Run autoregressive cross-model generation via Party A."""
        t_start = time.perf_counter()
        ctx = self._two_party_ctx()
        generated_text = ""
        num_generated = 0
        stop_reason = "max_tokens"

        for event in generate_two_party_stream(
            ctx, prompt, max_tokens=max_tokens, system_prompt=system_prompt,
        ):
            if event["type"] == "error":
                raise RuntimeError(event["message"])
            if event["type"] == "token":
                token_text = event["text"]
                generated_text += token_text
                num_generated = event["step"]
                if print_tokens:
                    sys.stdout.write(token_text)
                    sys.stdout.flush()
            elif event["type"] == "done":
                generated_text = event["full_text"]
                num_generated = event["total_tokens"]
                stop_reason = event["stop_reason"]

        if print_tokens:
            print()

        elapsed = time.perf_counter() - t_start
        tok_per_sec = num_generated / elapsed if elapsed > 0 else 0.0
        avg_step_ms = (elapsed / num_generated * 1000) if num_generated else 0.0

        return {
            "text": generated_text,
            "num_tokens": num_generated,
            "tokens_per_second": tok_per_sec,
            "elapsed_seconds": elapsed,
            "avg_step_ms": avg_step_ms,
            "stop_reason": stop_reason,
        }


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.client
# ---------------------------------------------------------------------------

def main():
    from src.model_catalog import build_runtime_config, get_active_ids

    config = load_config()
    pr_id, sp_id = get_active_ids(config)
    config = build_runtime_config(config, pr_id, sp_id)
    max_tokens = config["generation"].get("max_tokens", 100)

    parser = argparse.ArgumentParser(description="CrossMind client (Party B)")
    parser.add_argument("--server", default=None, help="Server URL (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=max_tokens)
    parser.add_argument(
        "--passphrase",
        default=None,
        help="Shared passphrase for rotation key (both parties derive locally)",
    )
    parser.add_argument(
        "--use-handshake",
        action="store_true",
        help="Send full rotation matrix via /handshake (legacy)",
    )
    args = parser.parse_args()

    prompts = [
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
    ]

    client = CrossMindClient(config, server_url=args.server)

    key_exchange = config.get("network", {}).get("key_exchange", "passphrase")
    if args.use_handshake or (args.passphrase is None and key_exchange == "handshake"):
        client.handshake()
    else:
        passphrase = args.passphrase or "hackathon2026"
        info = client.set_passphrase(passphrase)
        print(f"[client] Passphrase key: fingerprint={info['fingerprint']}")
        client.sync_passphrase_to_server(passphrase)
        print("[client] Specialist server key synced via passphrase")

    qwen_name = Path(config["models"]["qwen"]["model_id"]).name
    llama_name = Path(config["models"]["llama"]["model_id"]).name

    print("\n" + "=" * 60)
    print(f"  Cross-Model Generation  ({qwen_name} → {llama_name})")
    print(f"  Server: {client.server_url}")
    print(f"  max_tokens = {args.max_tokens}")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 60}")
        print(f"PROMPT {i}/{len(prompts)}:")
        print(f"  {prompt}\n")
        print("OUTPUT: ", end="")

        result = client.generate(prompt, max_tokens=args.max_tokens)

        print(
            f"\n  tokens={result['num_tokens']}  "
            f"{result['tokens_per_second']:.1f} tok/s  "
            f"avg_step={result['avg_step_ms']:.0f}ms  "
            f"stop={result['stop_reason']}"
        )

    print("\n" + "=" * 60)
    print("  Generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
