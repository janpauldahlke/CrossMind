"""
Rotation-based encryption for aligned hidden-state vectors.

An orthogonal matrix R (where R^T @ R = I) rotates vectors in
high-dimensional space without altering their magnitude.  This makes
intercepted vectors meaningless without the shared key, while allowing
lossless recovery via the transpose.

Key generation uses QR decomposition of a random matrix, which produces
a uniformly distributed orthogonal matrix in O(d^2) time.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch


def passphrase_to_seed(passphrase: str) -> int:
    """Derive a deterministic uint64 seed from a shared passphrase.

    Both parties call this with the same passphrase to obtain the same
    rotation matrix without transmitting the 64 MB key over the network.
    """
    digest = hashlib.sha256(passphrase.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def derive_rotation_key(dim: int, passphrase: str) -> np.ndarray:
    """Build rotation matrix R from a shared passphrase."""
    return generate_rotation_key(dim, seed=passphrase_to_seed(passphrase))


def key_fingerprint(R: np.ndarray) -> str:
    """Short hex fingerprint for out-of-band key alignment checks."""
    return hashlib.sha256(R.astype(np.float32).tobytes()).hexdigest()[:16]


def generate_rotation_key(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate an orthogonal rotation matrix via QR decomposition.

    Returns an (dim, dim) float32 orthogonal matrix R satisfying R^T @ R ≈ I.
    With a fixed *seed* the key is reproducible (useful for shared-seed
    key exchange without transmitting the 64 MB matrix).
    """
    rng = np.random.default_rng(seed)
    random_matrix = rng.standard_normal((dim, dim)).astype(np.float32)
    R, _ = np.linalg.qr(random_matrix)
    return R.astype(np.float32)


def encrypt(vector: torch.Tensor | np.ndarray, R: torch.Tensor) -> torch.Tensor:
    """Rotate *vector* by key *R*: ``v @ R``."""
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector).to(R.device)
    return vector @ R


def decrypt(vector: torch.Tensor | np.ndarray, R: torch.Tensor) -> torch.Tensor:
    """Recover original vector: ``v @ R.T``."""
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector).to(R.device)
    return vector @ R.T


def save_key(R: np.ndarray, path: str | Path) -> None:
    """Persist a rotation key to a ``.npy`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), R)


def load_key(path: str | Path) -> np.ndarray:
    """Load a rotation key from a ``.npy`` file."""
    return np.load(str(path)).astype(np.float32)


def verify_key(R: np.ndarray) -> dict:
    """Check that *R* is orthogonal (R^T @ R ≈ I).

    Returns a dict with ``max_error``, ``mean_error``, and a boolean
    ``is_valid`` (True when max error < 1e-5).
    """
    identity = np.eye(R.shape[0], dtype=np.float32)
    diff = R.T @ R - identity
    max_err = float(np.max(np.abs(diff)))
    mean_err = float(np.mean(np.abs(diff)))
    return {
        "max_error": max_err,
        "mean_error": mean_err,
        "is_valid": max_err < 1e-5,
    }
