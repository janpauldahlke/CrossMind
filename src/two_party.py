"""
Two-party cross-model generation over HTTP.

Party B (practitioner) encodes with Qwen, aligns, encrypts, and POSTs each
step to Party A (specialist) /infer.  Yields streaming events for WebSocket
UI and wire-visibility panels.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator

import httpx
import numpy as np
import torch

from src.generation import SYSTEM_PROMPT, _encode_continuation


def encode_vector_b64(v: np.ndarray) -> str:
    """float32 numpy vector → base64 string."""
    return base64.b64encode(v.astype(np.float32).tobytes()).decode()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def wire_metadata(vector_b64: str, server_url: str, step: int) -> dict:
    """Metadata for wire-visibility panels (no prompt text)."""
    raw_len = len(base64.b64decode(vector_b64))
    return {
        "url": f"{server_url.rstrip('/')}/infer",
        "method": "POST",
        "step": step,
        "bytes_on_wire": raw_len,
        "vector_b64_prefix": vector_b64[:48],
    }


@dataclass
class TwoPartyContext:
    """Runtime state for practitioner-side two-party generation."""

    qwen_model: torch.nn.Module
    qwen_tokenizer: object
    llama_tokenizer: object
    W_star: torch.Tensor
    b_star: torch.Tensor
    R_torch: torch.Tensor | None
    stop_ids: set[int]
    device: torch.device
    server_url: str
    http: httpx.Client
    hidden_dim: int
    lm_head: np.ndarray | None = None
    vector_sample_size: int = 64


def generate_two_party_stream(
    ctx: TwoPartyContext,
    prompt: str,
    max_tokens: int = 50,
    *,
    include_eavesdropper_preview: bool = False,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> Iterator[dict]:
    """Yield token/done/error events for practitioner UI and wire panels."""
    if ctx.R_torch is None:
        yield {"type": "error", "message": "Rotation key not set — enter passphrase first"}
        return

    if system_prompt and hasattr(ctx.qwen_tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = ctx.qwen_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        input_ids = ctx.qwen_tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(ctx.device)
    else:
        input_ids = ctx.qwen_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        )["input_ids"].to(ctx.device)

    generated_text = ""
    generated_ids: list[int] = []
    num_generated = 0
    past_key_values = None
    stop_reason = "max_tokens"
    t_start = time.perf_counter()

    with torch.no_grad():
        output = ctx.qwen_model.model(input_ids, use_cache=True)
        hidden = output.last_hidden_state
        past_key_values = output.past_key_values

        for step in range(max_tokens):
            h_B = hidden[:, -1:, :].float()
            h_aligned = h_B @ ctx.W_star + ctx.b_star
            h_enc = h_aligned @ ctx.R_torch

            h_enc_np = h_enc[0, 0].cpu().numpy()
            vector_b64 = encode_vector_b64(h_enc_np)

            try:
                resp = ctx.http.post(
                    f"{ctx.server_url.rstrip('/')}/infer",
                    json={"vector_b64": vector_b64, "step": step},
                )
                resp.raise_for_status()
                result = resp.json()
            except httpx.HTTPError as exc:
                yield {"type": "error", "message": f"Specialist request failed: {exc}"}
                return

            token_id = int(result["token_id"])
            token_text = result.get("token_text", "")

            if token_id in ctx.stop_ids:
                stop_reason = "eos"
                break

            if not token_text:
                new_full = ctx.llama_tokenizer.decode([token_id], skip_special_tokens=False)
                token_text = new_full

            generated_ids.append(token_id)
            new_full_text = ctx.llama_tokenizer.decode(
                generated_ids, skip_special_tokens=False,
            )
            token_text = new_full_text[len(generated_text):]

            if not token_text:
                continue

            generated_text = new_full_text
            num_generated += 1
            elapsed = time.perf_counter() - t_start
            tok_per_sec = num_generated / elapsed if elapsed > 0 else 0.0

            sample = h_enc_np[: ctx.vector_sample_size].tolist()
            yield {
                "type": "packet",
                "step": step,
                "encrypted_vector_sample": sample,
                "tok_per_sec": round(tok_per_sec, 1),
                "encryption_mode": "full",
                "wire_out": wire_metadata(vector_b64, ctx.server_url, step),
                "sent_at": _utc_now(),
            }

            new_qwen_ids = _encode_continuation(ctx.qwen_tokenizer, token_text)
            if not new_qwen_ids:
                continue

            new_input = torch.tensor(
                [new_qwen_ids], dtype=torch.long, device=ctx.device,
            )
            output = ctx.qwen_model.model(
                new_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = output.last_hidden_state
            past_key_values = output.past_key_values

    elapsed = time.perf_counter() - t_start
    avg_tok_sec = num_generated / elapsed if elapsed > 0 else 0.0
    done: dict = {
        "type": "done",
        "full_text": generated_text,
        "total_tokens": num_generated,
        "avg_tok_sec": round(avg_tok_sec, 1),
        "stop_reason": stop_reason,
    }
    yield done
