"""Platform helpers and config path resolution."""

from __future__ import annotations

import gc
import platform
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_NAME = "config.yaml"


def system() -> str:
    return platform.system()


def is_mac() -> bool:
    return system() == "Darwin"


def default_config_name() -> str:
    return CONFIG_NAME


def default_config_path(project_root: Path | None = None) -> Path:
    root = project_root or PROJECT_ROOT
    return root / default_config_name()


def empty_torch_cache() -> None:
    import torch

    gc.collect()
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def pair_storage_prefixes(config: dict) -> tuple[str, str | None]:
    """Return (primary pairs dir, optional fallback dir) from config."""
    storage = config.get("storage", {})
    primary = storage.get("pairs_dir", "data/pairs")
    fallback = storage.get("pairs_fallback")
    return primary, fallback
