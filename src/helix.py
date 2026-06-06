"""HELIX — Homomorphic Encryption on LInear head (CKKS).

Party B encrypts the aligned hidden state; Party A computes the routing
classification homomorphically (matmul + bias on ciphertext); Party B
decrypts the result.  Party A never sees plaintext vectors.

Usage
-----
    ctx = create_context()                          # Party B: keys
    pub_ctx = public_context(ctx)                   # send to Party A
    ct = encrypt_vector(ctx, h_aligned)             # Party B: encrypt

    ct_logits = classify_encrypted(pub_ctx, ct, W, b)   # Party A: blind compute
    logits = decrypt_vector(ctx, ct_logits)              # Party B: open result
    label  = int(np.argmax(logits))
"""

from __future__ import annotations

import base64
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tenseal as ts

CATEGORIES = ["Cardiology", "Neurology", "Oncology", "Orthopedics", "General_Medicine"]


@dataclass
class HelixResult:
    label: int
    label_name: str
    confidences: list[float]
    plaintext_matches: bool
    encrypt_ms: float
    compute_ms: float
    decrypt_ms: float
    total_ms: float
    ciphertext_sample: list[float]
    ciphertext_size_bytes: int


def create_context(
    poly_modulus_degree: int = 8192,
    coeff_mod_bit_sizes: list[int] | None = None,
    scale_power: int = 40,
) -> ts.Context:
    """Create a full CKKS context with secret + public + galois keys (Party B)."""
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 60]
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** scale_power
    return ctx


def public_context(ctx: ts.Context) -> ts.Context:
    """Derive a public-only context (no secret key) for Party A."""
    pub = ctx.copy()
    pub.make_context_public()
    return pub


def encrypt_vector(ctx: ts.Context, vector: np.ndarray) -> ts.CKKSVector:
    """Encrypt a 1-D float vector under CKKS (Party B)."""
    return ts.ckks_vector(ctx, vector.astype(np.float64).tolist())


def classify_encrypted(
    pub_ctx: ts.Context,
    ct: ts.CKKSVector,
    weights: np.ndarray,
    bias: np.ndarray,
) -> ts.CKKSVector:
    """Homomorphic routing: ct @ W + b  (Party A, no secret key)."""
    ct.link_context(pub_ctx)
    ct_logits = ct.matmul(weights.astype(np.float64).tolist())
    ct_logits += bias.astype(np.float64).tolist()
    return ct_logits


def extract_ciphertext_sample(ct: ts.CKKSVector, n: int = 64) -> tuple[list[float], int]:
    """Sample serialized CKKS bytes for UI visualization (what Party A sees)."""
    raw = ct.serialize()
    size = len(raw)
    if size == 0:
        return [0.0] * n, 0
    arr = np.frombuffer(raw, dtype=np.uint8)
    indices = np.linspace(0, len(arr) - 1, n, dtype=int)
    samples = (arr[indices].astype(np.float64) / 255.0).tolist()
    return samples, size


def decrypt_vector(ctx: ts.Context, ct: ts.CKKSVector) -> np.ndarray:
    """Decrypt a CKKS ciphertext back to a numpy array (Party B)."""
    ct.link_context(ctx)
    return np.array(ct.decrypt())


def classify_plaintext(
    vector: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Reference plaintext classification for comparison."""
    logits = vector.astype(np.float64) @ weights.astype(np.float64) + bias.astype(np.float64)
    return int(np.argmax(logits)), logits


def load_routing_head(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load routing head weights and bias from npz."""
    data = np.load(path)
    return data["weights"], data["bias"]


def helix_key_fingerprint(helix_key: str) -> str:
    """Short hex fingerprint for HELIX key alignment checks (UI only)."""
    digest = hashlib.sha256(helix_key.encode("utf-8")).digest()
    return digest.hex()[:16]


def serialize_context(ctx: ts.Context) -> bytes:
    return ctx.serialize()


def deserialize_context(raw: bytes) -> ts.Context:
    return ts.context_from(raw)


def serialize_vector(ct: ts.CKKSVector) -> bytes:
    return ct.serialize()


def deserialize_vector(ctx: ts.Context, raw: bytes) -> ts.CKKSVector:
    return ts.ckks_vector_from(ctx, raw)


def context_to_b64(ctx: ts.Context) -> str:
    return base64.b64encode(serialize_context(ctx)).decode("ascii")


def context_from_b64(b64: str) -> ts.Context:
    return deserialize_context(base64.b64decode(b64))


def vector_to_b64(ct: ts.CKKSVector) -> str:
    return base64.b64encode(serialize_vector(ct)).decode("ascii")


def vector_from_b64(ctx: ts.Context, b64: str) -> ts.CKKSVector:
    return deserialize_vector(ctx, base64.b64decode(b64))


def split_encrypt(
    ctx: ts.Context,
    h_aligned: np.ndarray,
) -> tuple[str, list[float], int, float]:
    """Party B: encrypt aligned vector; return b64 ciphertext + UI sample."""
    t0 = time.perf_counter()
    ct = encrypt_vector(ctx, h_aligned)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    sample, size_bytes = extract_ciphertext_sample(ct)
    return vector_to_b64(ct), sample, size_bytes, elapsed_ms


def split_compute(
    pub_ctx: ts.Context,
    ciphertext_b64: str,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[str, float, list[float], int]:
    """Party A: homomorphic classify; return b64 encrypted logits + UI sample."""
    t0 = time.perf_counter()
    ct = vector_from_b64(pub_ctx, ciphertext_b64)
    ct_logits = classify_encrypted(pub_ctx, ct, weights, bias)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    sample, size_bytes = extract_ciphertext_sample(ct_logits)
    return vector_to_b64(ct_logits), elapsed_ms, sample, size_bytes


def split_decrypt(
    ctx: ts.Context,
    logits_b64: str,
    h_aligned: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[int, list[float], float, bool]:
    """Party B: decrypt logits, compare with plaintext reference."""
    t0 = time.perf_counter()
    ct_logits = vector_from_b64(ctx, logits_b64)
    logits_dec = decrypt_vector(ctx, ct_logits)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    label_enc = int(np.argmax(logits_dec))
    label_plain, _ = classify_plaintext(h_aligned, weights, bias)
    probs = _softmax(logits_dec)
    confidences = [round(float(p), 4) for p in probs]
    return label_enc, confidences, elapsed_ms, label_enc == label_plain


def run_helix_classification(
    h_aligned: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    ctx: ts.Context | None = None,
) -> HelixResult:
    """Full HELIX pipeline: encrypt → homomorphic classify → decrypt → compare.

    If *ctx* is None a fresh context is created (adds ~100 ms one-time cost).
    """
    if ctx is None:
        ctx = create_context()
    pub_ctx = public_context(ctx)

    t0 = time.perf_counter()
    ct = encrypt_vector(ctx, h_aligned)
    t_encrypt = time.perf_counter()

    ciphertext_sample, ciphertext_size_bytes = extract_ciphertext_sample(ct)

    ct_logits = classify_encrypted(pub_ctx, ct, weights, bias)
    t_compute = time.perf_counter()

    logits_dec = decrypt_vector(ctx, ct_logits)
    t_decrypt = time.perf_counter()

    label_enc = int(np.argmax(logits_dec))
    label_plain, logits_plain = classify_plaintext(h_aligned, weights, bias)

    probs = _softmax(logits_dec)

    return HelixResult(
        label=label_enc,
        label_name=CATEGORIES[label_enc],
        confidences=[round(float(p), 4) for p in probs],
        plaintext_matches=(label_enc == label_plain),
        encrypt_ms=round((t_encrypt - t0) * 1000, 1),
        compute_ms=round((t_compute - t_encrypt) * 1000, 1),
        decrypt_ms=round((t_decrypt - t_compute) * 1000, 1),
        total_ms=round((t_decrypt - t0) * 1000, 1),
        ciphertext_sample=ciphertext_sample,
        ciphertext_size_bytes=ciphertext_size_bytes,
    )


def _softmax(x: np.ndarray, temperature: float = 3.0) -> np.ndarray:
    """Softmax with temperature scaling.  Higher temperature → flatter distribution."""
    scaled = (x - x.max()) / temperature
    exp = np.exp(scaled)
    return exp / exp.sum()
