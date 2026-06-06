"""Model catalog helpers for the web dashboard."""

from __future__ import annotations

import copy
from pathlib import Path

from src.platform import pair_storage_prefixes


def _entry_by_id(entries: list[dict], model_id: str) -> dict | None:
    return next((e for e in entries if e["id"] == model_id), None)


def _is_installed(entry: dict) -> bool:
    return Path(entry["model_path"]).is_dir()


def _pair_key(practitioner_id: str, specialist_id: str) -> str:
    return f"{practitioner_id}+{specialist_id}"


def _pair_layout(pair_id: str, config: dict | None = None) -> dict[str, str]:
    """Storage paths for a pair, honouring config storage overrides."""
    storage = (config or {}).get("storage", {})
    pairs_prefix = storage.get("pairs_dir", "data/pairs")
    extraction = (config or {}).get("extraction", {})
    raw_prefix = storage.get("raw_dir", extraction.get("raw_dir", "data/raw"))
    aligned_prefix = storage.get("aligned_dir", extraction.get("aligned_dir", "data/aligned"))
    pair_base = f"{pairs_prefix}/{pair_id}"
    return {
        "pair_dir": pair_base,
        "raw_practitioner_dir": f"{raw_prefix}/{pair_id}/practitioner",
        "raw_specialist_dir": f"{raw_prefix}/{pair_id}/specialist",
        "aligned_dir": f"{aligned_prefix}/{pair_id}",
        "map_path": f"{pair_base}/alignment_map.npz",
        "metrics_path": f"{pair_base}/alignment_metrics.json",
        "lm_head_path": f"{pair_base}/lm_head.npy",
        "tokenizer_path": f"{pair_base}/tokenizer",
    }


def enrich_pair_paths(pair: dict, config: dict | None = None) -> dict:
    """Ensure a pair dict has all storage paths for the canonical layout."""
    pair_id = pair.get("id") or _pair_key(pair["practitioner"], pair["specialist"])
    enriched = dict(pair)
    enriched.setdefault("id", pair_id)
    for key, value in _pair_layout(pair_id, config).items():
        enriched.setdefault(key, value)
    return enriched


_PATH_KEYS = (
    "raw_practitioner_dir",
    "raw_specialist_dir",
    "aligned_dir",
    "map_path",
    "metrics_path",
    "lm_head_path",
    "tokenizer_path",
)


def _path_populated(path: Path, key: str) -> bool:
    if key == "aligned_dir":
        return path.is_dir() and any(path.glob("chunk_*.npz"))
    if key in ("raw_practitioner_dir", "raw_specialist_dir"):
        return path.is_dir() and any(path.glob("*.npz"))
    if key == "tokenizer_path":
        return path.is_dir() and (
            (path / "tokenizer.json").exists()
            or (path / "tokenizer_config.json").exists()
        )
    return path.is_file()


def _path_candidates(config: dict, pair_id: str, key: str) -> list[str]:
    """Path candidates for a pair artifact key."""
    layout = _pair_layout(pair_id, config)
    return [layout[key]]


def resolve_pair_paths(
    config: dict,
    practitioner_id: str,
    specialist_id: str,
    project_root: Path | None = None,
) -> dict:
    """Resolve pair paths from config storage layout."""
    pair = resolve_pair(config, practitioner_id, specialist_id)
    root = project_root or Path(".")
    pair_id = pair["id"]
    layout = _pair_layout(pair_id, config)
    resolved = {**pair, **layout}

    for key in _PATH_KEYS:
        for candidate in _path_candidates(config, pair_id, key):
            if _path_populated(root / candidate, key):
                resolved[key] = candidate
                break
    return resolved


def resolve_pair(
    config: dict, practitioner_id: str, specialist_id: str
) -> dict:
    """Return pair config with paths, auto-generating for new combinations."""
    pair = find_pair(config, practitioner_id, specialist_id)
    if pair is None:
        pair = {
            "id": _pair_key(practitioner_id, specialist_id),
            "practitioner": practitioner_id,
            "specialist": specialist_id,
        }
    return enrich_pair_paths(pair, config)


def get_model_entry(config: dict, role: str, model_id: str) -> dict | None:
    catalog = config.get("model_catalog", {})
    key = "practitioners" if role == "practitioner" else "specialists"
    entry = _entry_by_id(catalog.get(key, []), model_id)
    if entry:
        return entry
    # Allow cross-list lookup for training flexibility
    other = "specialists" if key == "practitioners" else "practitioners"
    return _entry_by_id(catalog.get(other, []), model_id)


def training_catalog(config: dict) -> dict:
    """All installed models available for training role selection."""
    catalog = config.get("model_catalog", {})

    def _shape(entries: list[dict]) -> list[dict]:
        return [
            {
                "id": e["id"],
                "label": e["label"],
                "short_label": e.get("short_label", e["label"]),
                "hidden_dim": e["hidden_dim"],
                "installed": _is_installed(e),
            }
            for e in entries
        ]

    return {
        "practitioners": _shape(catalog.get("practitioners", [])),
        "specialists": _shape(catalog.get("specialists", [])),
    }


def pair_model_info(config: dict, practitioner_id: str, specialist_id: str) -> dict:
    practitioner = get_model_entry(config, "practitioner", practitioner_id)
    specialist = get_model_entry(config, "specialist", specialist_id)
    if not practitioner or not specialist:
        raise ValueError("Unknown practitioner or specialist model")
    return {
        "practitioner": practitioner["label"],
        "specialist": specialist["label"],
        "practitioner_short": practitioner.get("short_label", practitioner_id),
        "specialist_short": specialist.get("short_label", specialist_id),
        "practitioner_dim": practitioner["hidden_dim"],
        "specialist_dim": specialist["hidden_dim"],
        "alignment": f"{practitioner['hidden_dim']} → {specialist['hidden_dim']}",
        "practitioner_id": practitioner_id,
        "specialist_id": specialist_id,
    }


def find_pair(config: dict, practitioner_id: str, specialist_id: str) -> dict | None:
    for pair in config.get("model_pairs", []):
        if (
            pair["practitioner"] == practitioner_id
            and pair["specialist"] == specialist_id
        ):
            return pair
    return None


def discover_trained_pairs(project_root: Path) -> list[dict]:
    """Pairs with a trained alignment map under data/pairs/{pair_id}/."""
    pairs_dir = project_root / "data" / "pairs"
    if not pairs_dir.is_dir():
        return []

    found = []
    for entry in sorted(pairs_dir.iterdir()):
        if not entry.is_dir() or "+" not in entry.name:
            continue
        if not (entry / "alignment_map.npz").exists():
            continue
        practitioner_id, specialist_id = entry.name.split("+", 1)
        found.append(
            enrich_pair_paths(
                {
                    "id": entry.name,
                    "practitioner": practitioner_id,
                    "specialist": specialist_id,
                }
            )
        )
    return found


def iter_known_pairs(config: dict, project_root: Path) -> list[dict]:
    """Union of config model_pairs and pairs discovered on disk."""
    by_id: dict[str, dict] = {}
    for pair in config.get("model_pairs", []):
        enriched = enrich_pair_paths(pair, config)
        by_id[enriched["id"]] = enriched
    for pair in discover_trained_pairs(project_root):
        by_id.setdefault(pair["id"], pair)
    return list(by_id.values())


def _count_dataset(config: dict, project_root: Path) -> int:
    path = project_root / config["extraction"]["dataset_path"]
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def pair_extract_ready(
    config: dict, practitioner_id: str, specialist_id: str, project_root: Path
) -> bool:
    pair = resolve_pair_paths(config, practitioner_id, specialist_id, project_root)
    total = _count_dataset(config, project_root)
    if total == 0:
        return False
    pr_dir = project_root / pair["raw_practitioner_dir"]
    sp_dir = project_root / pair["raw_specialist_dir"]
    pr_count = len(list(pr_dir.glob("*.npz"))) if pr_dir.is_dir() else 0
    sp_count = len(list(sp_dir.glob("*.npz"))) if sp_dir.is_dir() else 0
    return pr_count >= total and sp_count >= total


def pair_align_ready(
    config: dict, practitioner_id: str, specialist_id: str, project_root: Path
) -> bool:
    pair = resolve_pair_paths(config, practitioner_id, specialist_id, project_root)
    return (
        _path_populated(project_root / pair["map_path"], "map_path")
        and _path_populated(project_root / pair["metrics_path"], "metrics_path")
    )


def pair_inference_ready(
    config: dict, practitioner_id: str, specialist_id: str, project_root: Path
) -> bool:
    if not pair_align_ready(config, practitioner_id, specialist_id, project_root):
        return False
    pair = resolve_pair_paths(config, practitioner_id, specialist_id, project_root)
    for key in ("lm_head_path", "tokenizer_path"):
        if not _path_populated(project_root / pair[key], key):
            return False
    practitioner = get_model_entry(config, "practitioner", practitioner_id)
    specialist = get_model_entry(config, "specialist", specialist_id)
    return bool(
        practitioner
        and specialist
        and _is_installed(practitioner)
        and _is_installed(specialist)
    )


def pair_status(
    config: dict, practitioner_id: str, specialist_id: str, project_root: Path
) -> dict:
    return {
        "extract": pair_extract_ready(
            config, practitioner_id, specialist_id, project_root
        ),
        "align": pair_align_ready(config, practitioner_id, specialist_id, project_root),
        "inference_ready": pair_inference_ready(
            config, practitioner_id, specialist_id, project_root
        ),
    }


def get_active_ids(config: dict) -> tuple[str, str]:
    active = config.get("active_models", {})
    practitioner_id = active.get("practitioner")
    specialist_id = active.get("specialist")

    if not practitioner_id or not specialist_id:
        practitioner_id = "qwen-2.5-7b"
        specialist_id = "llama-3.1-8b"

    return practitioner_id, specialist_id


def build_runtime_config(
    base_config: dict, practitioner_id: str, specialist_id: str
) -> dict:
    """Build a generation-ready config for a practitioner/specialist pair."""
    catalog = base_config.get("model_catalog")
    if not catalog:
        return base_config

    practitioner = get_model_entry(base_config, "practitioner", practitioner_id)
    specialist = get_model_entry(base_config, "specialist", specialist_id)
    pair = resolve_pair_paths(base_config, practitioner_id, specialist_id)

    if practitioner is None:
        raise ValueError(f"Unknown practitioner model: {practitioner_id}")
    if specialist is None:
        raise ValueError(f"Unknown specialist model: {specialist_id}")
    if not _is_installed(practitioner):
        raise ValueError(f"Practitioner model not installed: {practitioner['label']}")
    if not _is_installed(specialist):
        raise ValueError(f"Specialist model not installed: {specialist['label']}")

    config = copy.deepcopy(base_config)
    config["models"]["qwen"] = {
        "model_id": practitioner["model_path"],
        "hidden_dim": practitioner["hidden_dim"],
        "vocab_size": practitioner["vocab_size"],
        "quantization": practitioner.get("quantization", "none"),
    }
    config["models"]["llama"] = {
        "model_id": specialist["model_path"],
        "hidden_dim": specialist["hidden_dim"],
        "vocab_size": specialist["vocab_size"],
        "quantization": specialist.get("quantization", "none"),
    }
    config["alignment"]["map_path"] = pair["map_path"]
    config["alignment"]["metrics_path"] = pair["metrics_path"]
    config["generation"]["lm_head_path"] = pair["lm_head_path"]
    config["generation"]["tokenizer_path"] = pair["tokenizer_path"]
    config["active_models"] = {
        "practitioner": practitioner_id,
        "specialist": specialist_id,
    }
    return config


def aligned_specialists(
    config: dict, practitioner_id: str, project_root: Path | None = None
) -> set[str]:
    root = project_root or Path(".")
    return {
        pair["specialist"]
        for pair in iter_known_pairs(config, root)
        if pair["practitioner"] == practitioner_id
        and pair_inference_ready(config, pair["practitioner"], pair["specialist"], root)
    }


def aligned_practitioners(
    config: dict, specialist_id: str, project_root: Path | None = None
) -> set[str]:
    root = project_root or Path(".")
    return {
        pair["practitioner"]
        for pair in iter_known_pairs(config, root)
        if pair["specialist"] == specialist_id
        and pair_inference_ready(config, pair["practitioner"], pair["specialist"], root)
    }


def catalog_response(
    config: dict,
    metrics: dict | None = None,
    project_root: Path | None = None,
) -> dict:
    """Shape the model catalog for the frontend."""
    root = project_root or Path(".")
    catalog = config.get("model_catalog", {})
    practitioner_id, specialist_id = get_active_ids(config)
    status = pair_status(config, practitioner_id, specialist_id, root)
    metrics = metrics or {}

    practitioners = []
    for entry in catalog.get("practitioners", []):
        aligned = sorted(aligned_specialists(config, entry["id"], root))
        practitioners.append({
            "id": entry["id"],
            "label": entry["label"],
            "short_label": entry.get("short_label", entry["label"]),
            "installed": _is_installed(entry),
            "aligned_with": aligned,
        })

    specialists = []
    for entry in catalog.get("specialists", []):
        aligned = sorted(aligned_practitioners(config, entry["id"], root))
        specialists.append({
            "id": entry["id"],
            "label": entry["label"],
            "short_label": entry.get("short_label", entry["label"]),
            "installed": _is_installed(entry),
            "aligned_with": aligned,
        })

    practitioner = _entry_by_id(catalog.get("practitioners", []), practitioner_id)
    specialist = _entry_by_id(catalog.get("specialists", []), specialist_id)

    return {
        "active": {
            "practitioner": practitioner_id,
            "specialist": specialist_id,
        },
        "practitioners": practitioners,
        "specialists": specialists,
        "pair_ready": status["inference_ready"],
        "pair_status": status,
        "active_labels": {
            "practitioner": practitioner["label"] if practitioner else practitioner_id,
            "specialist": specialist["label"] if specialist else specialist_id,
            "practitioner_short": practitioner.get("short_label", practitioner_id) if practitioner else practitioner_id,
            "specialist_short": specialist.get("short_label", specialist_id) if specialist else specialist_id,
        },
        "metrics": {
            "cosine_similarity": metrics.get("cosine_similarity_mean"),
            "cka": metrics.get("cka"),
        },
    }


def default_catalog_from_legacy(config: dict) -> dict:
    """Build a catalog from legacy models.qwen / models.llama keys."""
    qwen = config["models"]["qwen"]
    llama = config["models"]["llama"]
    return {
        "practitioners": [{
            "id": "qwen-2.5-7b",
            "label": Path(qwen["model_id"]).name.replace("-", " "),
            "short_label": "Qwen 2.5",
            "model_path": qwen["model_id"],
            "hidden_dim": qwen["hidden_dim"],
            "vocab_size": qwen["vocab_size"],
            "quantization": qwen.get("quantization", "none"),
        }],
        "specialists": [{
            "id": "llama-3.1-8b",
            "label": Path(llama["model_id"]).name.replace("-", " "),
            "short_label": "Llama 3.1",
            "model_path": llama["model_id"],
            "hidden_dim": llama["hidden_dim"],
            "vocab_size": llama["vocab_size"],
            "quantization": llama.get("quantization", "none"),
        }],
    }
