"""
Cross-model text generation.

Uses Qwen's hidden states, transformed through the learned alignment map
(W*, b*), and decoded by Llama's LM head to produce text.  This validates
that the alignment learned in Phase 2 captures enough structure for
autoregressive generation across model boundaries.

Pipeline per step:
    Qwen inner model → last hidden state → W*·h + b* → Llama LM head → argmax
    → decode (Llama tokenizer) → re-encode (Qwen tokenizer) → next step

"""

import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.models import load_model, unload_model


def _encode_continuation(tokenizer, text: str) -> list[int]:
    """Encode *text* as a mid-sentence continuation, not as start-of-text.

    SentencePiece-based tokenizers (Phi, Mistral, LLaMA 2) auto-prepend a
    word-boundary marker (▁) to the first token.  When the decoded specialist
    token already starts with a space (e.g. " or"), the prefix doubles it into
    an extra ▁ token that corrupts the practitioner's KV cache.

    By encoding with a sentinel prefix and stripping the sentinel token IDs,
    the text is tokenized as a mid-sentence continuation — preserving its
    leading space exactly once.  For tiktoken-based tokenizers (Qwen, Llama 3)
    this is a no-op: the results are identical to a plain encode.
    """
    _SENTINEL = "."
    sentinel_ids = tokenizer.encode(_SENTINEL, add_special_tokens=False)
    full_ids = tokenizer.encode(_SENTINEL + text, add_special_tokens=False)
    if len(full_ids) > len(sentinel_ids):
        return full_ids[len(sentinel_ids):]
    return tokenizer.encode(text, add_special_tokens=False)


@dataclass
class GenerationComponents:
    """All components needed for cross-model generation."""

    qwen_model: object
    qwen_tokenizer: object
    llama_lm_head: torch.Tensor   # (vocab_size, hidden_dim_a) on GPU
    llama_tokenizer: object
    W_star: torch.Tensor           # (hidden_dim_b, hidden_dim_a) on GPU
    b_star: torch.Tensor           # (hidden_dim_a,) on GPU
    rotation_key: torch.Tensor | None = None  # (hidden_dim_a, hidden_dim_a) on GPU


@dataclass
class GenerationResult:
    """Output from a single cross-model generation call."""

    text: str
    num_tokens: int
    tokens_per_second: float
    stop_reason: str


# ---------------------------------------------------------------------------
# LM head extraction
# ---------------------------------------------------------------------------


def extract_lm_head(
    model_id: str, save_path: str, tokenizer_save_path: str
) -> None:
    """Load Llama, extract its LM head weight and tokenizer, then unload.

    The LM head is saved as a float32 numpy array of shape
    (vocab_size, hidden_dim).  The tokenizer is persisted via
    ``save_pretrained`` so it can be loaded independently later.

    """
    save_path = Path(save_path)
    tokenizer_save_path = Path(tokenizer_save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading Llama from {model_id} ...")
    model, tokenizer = load_model(model_id)

    if hasattr(model, "lm_head"):
        lm_head_weight = model.lm_head.weight.data.cpu().float().numpy()
    else:
        lm_head_weight = model.model.embed_tokens.weight.data.cpu().float().numpy()

    print(f"LM head shape: {lm_head_weight.shape}, dtype: {lm_head_weight.dtype}")
    np.save(str(save_path), lm_head_weight)
    print(f"Saved LM head → {save_path}")

    tokenizer.save_pretrained(str(tokenizer_save_path))
    print(f"Saved tokenizer → {tokenizer_save_path}")

    from src.platform import empty_torch_cache

    unload_model(model)
    del tokenizer
    empty_torch_cache()


# ---------------------------------------------------------------------------
# Component loading
# ---------------------------------------------------------------------------


def load_generation_components(config: dict) -> GenerationComponents:
    """Load all components needed for cross-model generation.

    Returns a ``GenerationComponents`` dataclass with:
    - Qwen model (float16, for forward passes)
    - Qwen tokenizer
    - W_star / b_star alignment tensors (float32 on GPU)
    - Llama LM head weight (float32 on GPU)
    - Llama tokenizer (for decoding generated token IDs)
    """
    qwen_cfg = config["models"]["qwen"]
    gen_cfg = config["generation"]
    map_path = config["alignment"]["map_path"]

    print("Loading Qwen model ...")
    qwen_model, qwen_tokenizer = load_model(qwen_cfg["model_id"])
    device = next(qwen_model.parameters()).device

    print(f"Loading alignment map from {map_path} ...")
    alignment = np.load(map_path)
    W_star = torch.from_numpy(alignment["W_star"].astype(np.float32)).to(device)
    b_star = torch.from_numpy(alignment["b_star"].astype(np.float32)).to(device)
    print(f"  W* {tuple(W_star.shape)}  b* {tuple(b_star.shape)}")

    lm_head_path = gen_cfg["lm_head_path"]
    print(f"Loading Llama LM head from {lm_head_path} ...")
    lm_head_np = np.load(lm_head_path)
    llama_lm_head = torch.from_numpy(lm_head_np.astype(np.float32)).to(device)
    print(f"  LM head {tuple(llama_lm_head.shape)}")

    tok_path = gen_cfg["tokenizer_path"]
    print(f"Loading Llama tokenizer from {tok_path} ...")
    llama_tokenizer = AutoTokenizer.from_pretrained(
        tok_path, clean_up_tokenization_spaces=False,
    )

    return GenerationComponents(
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        llama_lm_head=llama_lm_head,
        llama_tokenizer=llama_tokenizer,
        W_star=W_star,
        b_star=b_star,
    )


def load_generation_components_encrypted(config: dict) -> GenerationComponents:
    """Load generation components with the rotation key from config.

    Delegates to :func:`load_generation_components` and then loads the
    rotation key specified in ``config["encryption"]["key_path"]``,
    attaching it to the returned components as a float32 GPU tensor.
    """
    from src.encryption import load_key

    components = load_generation_components(config)

    key_path = config["encryption"]["key_path"]
    print(f"Loading rotation key from {key_path} ...")
    R_np = load_key(key_path)
    device = components.W_star.device
    components.rotation_key = torch.from_numpy(R_np).to(device)
    print(f"  Rotation key {tuple(components.rotation_key.shape)}")

    return components


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _get_stop_token_ids(tokenizer) -> set[int]:
    """Collect all stop / end-of-turn token IDs from the specialist tokenizer.

    Covers Llama (<|eot_id|>), Mistral (</s>, [/INST]), Qwen (<|im_end|>),
    and any model whose tokenizer exposes eos_token_id.
    """
    stop_ids: set[int] = set()

    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)

    unk_id = tokenizer.unk_token_id
    _STOP_NAMES = [
        "<|eot_id|>",       # Llama 3
        "<|end_of_text|>",  # Llama 3
        "</s>",             # Mistral / SentencePiece convention
        "<|im_end|>",       # Qwen / ChatML
        "<|endoftext|>",    # GPT-NeoX family
    ]
    for name in _STOP_NAMES:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and (unk_id is None or tid != unk_id):
            stop_ids.add(tid)

    return stop_ids


SYSTEM_PROMPT = (
    "You are a medical specialist providing brief clinical assessments. "
    "Give a concise diagnosis or recommendation in 1-2 sentences. "
    "Do not use multiple choice format. Do not list options. "
    "Respond directly with your clinical assessment."
)


def generate_cross_model(
    prompt: str,
    components: GenerationComponents,
    max_tokens: int = 50,
    encryption_mode: str = "none",
    wrong_key: torch.Tensor | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> GenerationResult:
    """Autoregressive cross-model generation.

    At each step:
        1. Forward new token(s) through Qwen's inner model (with KV cache).
        2. Take the last-position hidden state.
        3. Transform via the alignment map: ``h_aligned = h @ W* + b*``.
        4. Optionally apply rotation encryption / decryption.
        5. Compute logits: ``logits = h_aligned @ llama_lm_head.T``.
        6. Greedy-decode the argmax token with Llama's tokenizer.
        7. Re-encode the decoded text with Qwen's tokenizer (no special
           tokens) and feed back for the next step.

    When *system_prompt* is provided, the input is formatted using the
    tokenizer's chat template (instruction-following mode).

    *encryption_mode* controls the rotation layer:

    - ``"none"`` -- Phase 3 behaviour, no encryption (default).
    - ``"full"`` -- encrypt with R then decrypt with R^T (lossless roundtrip).
    - ``"no_decrypt"`` -- encrypt only; simulates an eavesdropper that
      intercepts the ciphertext without the key.
    - ``"wrong_key"`` -- encrypt with R, decrypt with a different key
      supplied via *wrong_key*.

    Handles multi-token re-encoding (a single Llama token may expand to
    several Qwen tokens) and skips tokens that decode to empty strings.
    """
    qwen_model = components.qwen_model
    qwen_tokenizer = components.qwen_tokenizer
    llama_lm_head = components.llama_lm_head
    llama_tokenizer = components.llama_tokenizer
    W_star = components.W_star
    b_star = components.b_star
    R = components.rotation_key

    if encryption_mode != "none" and R is None:
        raise ValueError(
            f"encryption_mode={encryption_mode!r} requires a rotation_key "
            "in GenerationComponents"
        )

    stop_ids = _get_stop_token_ids(llama_tokenizer)
    device = next(qwen_model.parameters()).device

    if system_prompt and hasattr(qwen_tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = qwen_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        input_ids = qwen_tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(device)
    else:
        input_ids = qwen_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        )["input_ids"].to(device)

    generated_text = ""
    generated_ids: list[int] = []
    num_generated = 0
    past_key_values = None
    stop_reason = "max_tokens"

    t_start = time.perf_counter()

    with torch.no_grad():
        output = qwen_model.model(input_ids, use_cache=True)
        hidden = output.last_hidden_state
        past_key_values = output.past_key_values

        for _ in range(max_tokens):
            h_B = hidden[:, -1:, :].float()

            h_aligned = h_B @ W_star + b_star

            if encryption_mode == "full":
                h_aligned = h_aligned @ R
                h_aligned = h_aligned @ R.T
            elif encryption_mode == "no_decrypt":
                h_aligned = h_aligned @ R
            elif encryption_mode == "wrong_key":
                h_aligned = h_aligned @ R
                h_aligned = h_aligned @ wrong_key.T

            logits = h_aligned @ llama_lm_head.T
            next_token_id = int(torch.argmax(logits[0, 0]).item())

            if next_token_id in stop_ids:
                stop_reason = "eos"
                break

            generated_ids.append(next_token_id)
            new_full_text = llama_tokenizer.decode(
                generated_ids, skip_special_tokens=False,
            )
            token_text = new_full_text[len(generated_text):]

            if not token_text:
                continue

            generated_text = new_full_text
            num_generated += 1

            new_qwen_ids = _encode_continuation(qwen_tokenizer, token_text)
            if not new_qwen_ids:
                continue

            new_input = torch.tensor(
                [new_qwen_ids], dtype=torch.long, device=device,
            )

            output = qwen_model.model(
                new_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = output.last_hidden_state
            past_key_values = output.past_key_values

    elapsed = time.perf_counter() - t_start
    tok_per_sec = num_generated / elapsed if elapsed > 0 else 0.0

    return GenerationResult(
        text=generated_text,
        num_tokens=num_generated,
        tokens_per_second=tok_per_sec,
        stop_reason=stop_reason,
    )


def generate_cross_model_stream(
    prompt: str,
    components: GenerationComponents,
    max_tokens: int = 50,
    encryption_mode: str = "none",
    wrong_key: torch.Tensor | None = None,
    system_prompt: str | None = SYSTEM_PROMPT,
    vector_sample_size: int = 64,
):
    """Streaming variant of :func:`generate_cross_model`.

    Yields one dict per generated token (plus a final ``done`` dict):

    - ``{"type": "token", "text", "token_id", "step", "vector_sample", ...}``
    - ``{"type": "done", "full_text", "total_tokens", "avg_tok_sec", ...}``
    """
    qwen_model = components.qwen_model
    qwen_tokenizer = components.qwen_tokenizer
    llama_lm_head = components.llama_lm_head
    llama_tokenizer = components.llama_tokenizer
    W_star = components.W_star
    b_star = components.b_star
    R = components.rotation_key

    if encryption_mode != "none" and R is None:
        raise ValueError(
            f"encryption_mode={encryption_mode!r} requires a rotation_key "
            "in GenerationComponents"
        )

    stop_ids = _get_stop_token_ids(llama_tokenizer)
    device = next(qwen_model.parameters()).device

    if system_prompt and hasattr(qwen_tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = qwen_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        input_ids = qwen_tokenizer(
            formatted, return_tensors="pt", add_special_tokens=False,
        )["input_ids"].to(device)
    else:
        input_ids = qwen_tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
        )["input_ids"].to(device)

    generated_text = ""
    generated_ids: list[int] = []
    intercepted_text = ""
    intercepted_ids: list[int] = []
    num_generated = 0
    past_key_values = None
    stop_reason = "max_tokens"
    t_start = time.perf_counter()

    with torch.no_grad():
        output = qwen_model.model(input_ids, use_cache=True)
        hidden = output.last_hidden_state
        past_key_values = output.past_key_values

        for step in range(max_tokens):
            h_B = hidden[:, -1:, :].float()
            h_aligned = h_B @ W_star + b_star

            vector_for_preview = h_aligned[0, 0].cpu().numpy()
            h_for_lm = h_aligned
            intercepted_token_text = None

            if encryption_mode == "full":
                h_enc = h_aligned @ R
                vector_for_preview = h_enc[0, 0].cpu().numpy()
                h_for_lm = h_enc @ R.T

                intercepted_logits = h_enc @ llama_lm_head.T
                intercepted_id = int(torch.argmax(intercepted_logits[0, 0]).item())
                intercepted_ids.append(intercepted_id)
                new_intercepted = llama_tokenizer.decode(
                    intercepted_ids, skip_special_tokens=False,
                )
                intercepted_token_text = new_intercepted[len(intercepted_text):]
                intercepted_text = new_intercepted
            elif encryption_mode == "no_decrypt":
                h_enc = h_aligned @ R
                vector_for_preview = h_enc[0, 0].cpu().numpy()
                h_for_lm = h_enc
            elif encryption_mode == "wrong_key":
                h_enc = h_aligned @ R
                vector_for_preview = h_enc[0, 0].cpu().numpy()
                h_for_lm = h_enc @ wrong_key.T

            logits = h_for_lm @ llama_lm_head.T
            next_token_id = int(torch.argmax(logits[0, 0]).item())

            if next_token_id in stop_ids:
                stop_reason = "eos"
                break

            generated_ids.append(next_token_id)
            new_full_text = llama_tokenizer.decode(
                generated_ids, skip_special_tokens=False,
            )
            token_text = new_full_text[len(generated_text):]

            if not token_text:
                continue

            generated_text = new_full_text
            num_generated += 1
            elapsed = time.perf_counter() - t_start
            tok_per_sec = num_generated / elapsed if elapsed > 0 else 0.0

            sample = vector_for_preview[:vector_sample_size].tolist()
            event: dict = {
                "type": "token",
                "text": token_text,
                "token_id": next_token_id,
                "step": num_generated,
                "encrypted_vector_sample": sample,
                "tok_per_sec": round(tok_per_sec, 1),
                "encryption_mode": encryption_mode,
            }
            if intercepted_token_text is not None:
                event["intercepted_text"] = intercepted_token_text
            yield event

            new_qwen_ids = _encode_continuation(qwen_tokenizer, token_text)
            if not new_qwen_ids:
                continue

            new_input = torch.tensor(
                [new_qwen_ids], dtype=torch.long, device=device,
            )
            output = qwen_model.model(
                new_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = output.last_hidden_state
            past_key_values = output.past_key_values

    elapsed = time.perf_counter() - t_start
    avg_tok_sec = num_generated / elapsed if elapsed > 0 else 0.0

    done_event: dict = {
        "type": "done",
        "full_text": generated_text,
        "total_tokens": num_generated,
        "avg_tok_sec": round(avg_tok_sec, 1),
        "stop_reason": stop_reason,
    }
    if intercepted_text:
        done_event["intercepted_full_text"] = intercepted_text
    yield done_event
