"""
Model loading and hidden state extraction.

Loads models as float16 via MPS (Apple Silicon).
"""

import gc
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.platform import default_config_path, empty_torch_cache


def load_config(path: str | None = None) -> dict:
    """Load config; defaults to config.yaml."""
    config_path = Path(path) if path else default_config_path()
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_id: str, quantize: bool = False):
    """Load a model and tokenizer, returning (model, tokenizer).

    Models are loaded as float16 full weights via MPS.
    The *quantize* parameter is accepted for call-site compatibility
    but has no effect.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    return model, tokenizer


def unload_model(model):
    """Delete model and free accelerator memory."""
    del model
    empty_torch_cache()


def get_token_offsets(
    tokenizer, text: str, token_ids: list[int]
) -> list[tuple[int, int]]:
    """Get character-level (start, end) offsets for each token.

    Uses the HuggingFace fast tokenizer's offset mapping when available,
    falls back to manual reconstruction otherwise.
    Returns offsets only for content tokens (BOS/EOS stripped).
    """
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    return [(s, e) for s, e in offsets if e > s or s == e == 0]


def extract_hidden_states(
    model, tokenizer, text: str
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract final hidden states (post-norm, pre-LM-head) for a text.

    Returns:
        hidden_states: shape (seq_len, hidden_dim), float16
        token_offsets: list of (char_start, char_end) for content tokens
    """
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        base_output = model.model(input_ids)
        hidden = base_output.last_hidden_state  # (1, seq_len, hidden_dim)

    hidden_np = hidden[0].cpu().to(torch.float16).numpy()  # (seq_len, hidden_dim)

    # Get offsets for content tokens only (no BOS/EOS)
    content_ids = input_ids[0].tolist()
    offsets = get_token_offsets(tokenizer, text, content_ids)

    # Strip BOS/EOS from hidden states to match content offsets.
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    start_idx = 0
    end_idx = hidden_np.shape[0]

    if bos_id is not None and len(content_ids) > 0 and content_ids[0] == bos_id:
        start_idx = 1
    if eos_id is not None and len(content_ids) > 0 and content_ids[-1] == eos_id:
        end_idx -= 1

    hidden_np = hidden_np[start_idx:end_idx]

    # Ensure lengths match: truncate to the shorter length.
    min_len = min(len(offsets), hidden_np.shape[0])
    hidden_np = hidden_np[:min_len]
    offsets = offsets[:min_len]

    return hidden_np, offsets
