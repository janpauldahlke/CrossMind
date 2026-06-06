"""
Practitioner (Party B) API for the split demo.

Loads Qwen + alignment map, derives rotation key from passphrase, and streams
cross-party generation to the clinic UI while POSTing encrypted vectors to the
specialist server.
"""

import asyncio
import gc
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.encryption import derive_rotation_key, key_fingerprint, verify_key
from src.two_party import TwoPartyContext, generate_two_party_stream

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MEDICAL_PHRASES = [
    "Patient presents with persistent cough, night sweats, and weight loss over 3 months.",
    "A 45-year-old male with chest pain radiating to the left arm and shortness of breath.",
    "Recommend treatment for Type 2 diabetes with comorbid chronic kidney disease.",
    "30-year-old female with fatigue, weight gain, and cold intolerance.",
    "Explain the mechanism of action of metformin for glucose management.",
    "Differential diagnosis for sudden onset severe headache with neck stiffness.",
    "Child presents with high fever, rash, and strawberry tongue.",
    "Post-operative patient with sudden dyspnea and pleuritic chest pain.",
]

_state: dict = {
    "config": None,
    "ctx": None,
    "max_tokens": 50,
    "specialist_url": None,
    "generating": False,
    "helix_classifying": False,
    "loading": False,
    "metrics": None,
    "key_fingerprint": None,
    "helix_ctx": None,
    "helix_key_fingerprint": None,
    "helix_routing_head": None,
}


def _load_config() -> dict:
    from src.models import load_config
    from src.model_catalog import default_catalog_from_legacy

    config = load_config()
    if "model_catalog" not in config:
        config["model_catalog"] = default_catalog_from_legacy(config)
    return config


def _load_metrics(config: dict) -> dict:
    metrics_path = Path(config["alignment"]["metrics_path"])
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def _get_stop_token_ids(tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    unk_id = tokenizer.unk_token_id
    for name in [
        "<|eot_id|>", "<|end_of_text|>",
        "</s>",
        "<|im_end|>",
        "<|endoftext|>",
    ]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid is not None and (unk_id is None or tid != unk_id):
            stop_ids.add(tid)
    return stop_ids


def _load_practitioner_sync(config: dict) -> TwoPartyContext:
    import httpx
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from src.model_catalog import build_runtime_config, get_active_ids, pair_inference_ready
    from src.models import load_model

    pr_id, sp_id = get_active_ids(config)
    runtime = build_runtime_config(config, pr_id, sp_id)

    if not pair_inference_ready(runtime, pr_id, sp_id, PROJECT_ROOT):
        raise RuntimeError(
            "Model pair not ready for inference — run extraction and alignment first."
        )

    qwen_cfg = runtime["models"]["qwen"]
    map_path = runtime["alignment"]["map_path"]
    gen_cfg = runtime["generation"]
    enc_cfg = runtime["encryption"]

    print("[practitioner_api] Loading Qwen model ...")
    qwen_model, qwen_tokenizer = load_model(qwen_cfg["model_id"])
    device = next(qwen_model.parameters()).device

    print(f"[practitioner_api] Loading alignment map from {map_path} ...")
    alignment = np.load(map_path)
    W_star = torch.from_numpy(alignment["W_star"].astype(np.float32)).to(device)
    b_star = torch.from_numpy(alignment["b_star"].astype(np.float32)).to(device)

    tok_path = gen_cfg["tokenizer_path"]
    llama_tokenizer = AutoTokenizer.from_pretrained(
        tok_path, clean_up_tokenization_spaces=False,
    )

    lm_head = None
    lm_head_path = gen_cfg.get("lm_head_path")
    if lm_head_path and Path(lm_head_path).exists():
        lm_head = np.load(lm_head_path).astype(np.float32)
        print(f"[practitioner_api] LM head loaded for eavesdropper preview {lm_head.shape}")

    specialist_url = _state["specialist_url"] or runtime["network"]["party_b"]["server_url"]

    return TwoPartyContext(
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
        llama_tokenizer=llama_tokenizer,
        W_star=W_star,
        b_star=b_star,
        R_torch=None,  # type: ignore[arg-type]
        stop_ids=_get_stop_token_ids(llama_tokenizer),
        device=device,
        server_url=specialist_url,
        http=httpx.Client(timeout=60.0),
        hidden_dim=enc_cfg["key_dim"],
        lm_head=lm_head,
    )


def _unload_practitioner() -> None:
    from src.models import unload_model

    ctx: TwoPartyContext | None = _state.get("ctx")
    if ctx is None:
        return
    if ctx.http is not None:
        ctx.http.close()
    unload_model(ctx.qwen_model)
    _state["ctx"] = None
    del ctx
    gc.collect()


def _config_payload() -> dict:
    from src.model_catalog import catalog_response, get_active_ids, pair_inference_ready

    config = _state["config"]
    payload = catalog_response(config, _state["metrics"], PROJECT_ROOT)
    pr_id, sp_id = get_active_ids(config)
    payload["ready"] = _state["ctx"] is not None and not _state["loading"]
    payload["loading"] = _state["loading"]
    payload["max_tokens"] = _state["max_tokens"]
    payload["specialist_url"] = _state["specialist_url"]
    payload["key_set"] = _state["key_fingerprint"] is not None
    payload["key_fingerprint"] = _state["key_fingerprint"]
    payload["pair_ready"] = pair_inference_ready(config, pr_id, sp_id, PROJECT_ROOT)
    payload["role"] = "practitioner"
    payload["helix_available"] = _state["helix_routing_head"] is not None
    helix_active = _state["helix_ctx"] is not None
    payload["helix_bootstrapped"] = helix_active
    payload["helix_key_fingerprint"] = _state["helix_key_fingerprint"] if helix_active else None
    sync = _helix_sync_status()
    payload["helix_hospital_fingerprint"] = sync["hospital_fingerprint"]
    payload["helix_hospital_synced"] = sync["synced"]
    return payload


def _hospital_clear_helix() -> None:
    import httpx

    url = _state["specialist_url"]
    if not url:
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(f"{url.rstrip('/')}/api/helix/clear")
    except Exception as exc:
        print(f"[practitioner_api] HELIX clear on hospital failed: {exc}")


def _hospital_bootstrap_helix(ctx, fingerprint: str) -> None:
    import httpx

    from src.helix import context_to_b64, public_context

    url = _state["specialist_url"]
    if not url:
        raise RuntimeError("No specialist URL configured")

    pub_b64 = context_to_b64(public_context(ctx))
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{url.rstrip('/')}/api/helix/bootstrap",
            json={"public_ctx_b64": pub_b64, "fingerprint": fingerprint},
        )
        resp.raise_for_status()


def _fetch_hospital_helix_status() -> dict | None:
    """Return hospital HELIX status, or None if unreachable."""
    url = _state["specialist_url"]
    if not url:
        return None

    import httpx

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{url.rstrip('/')}/api/helix/status")
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None


def _helix_sync_status() -> dict:
    """Compare clinic HELIX session with hospital bootstrap state."""
    clinic_fp = _state["helix_key_fingerprint"]
    remote = _fetch_hospital_helix_status()
    hospital_fp = remote.get("fingerprint") if remote and remote.get("bootstrapped") else None
    synced = (
        clinic_fp is not None
        and hospital_fp is not None
        and clinic_fp == hospital_fp
    )
    return {
        "clinic_fingerprint": clinic_fp,
        "hospital_fingerprint": hospital_fp,
        "synced": synced,
    }


def _clear_helix_session() -> None:
    _state["helix_ctx"] = None
    _state["helix_key_fingerprint"] = None
    _hospital_clear_helix()


def _set_passphrase(passphrase: str) -> dict:
    ctx: TwoPartyContext | None = _state["ctx"]
    if ctx is None:
        raise HTTPException(status_code=503, detail="Practitioner not ready")

    import torch

    R = derive_rotation_key(ctx.hidden_dim, passphrase)
    stats = verify_key(R)
    ctx.R_torch = torch.from_numpy(R).to(ctx.device)
    fp = key_fingerprint(R)
    _state["key_fingerprint"] = fp
    print(f"[practitioner_api] Passphrase set: fingerprint={fp}")
    return {
        "status": "ok",
        "fingerprint": fp,
        "orthogonality_ok": stats["is_valid"],
        "max_error": stats["max_error"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = _load_config()
    _state["config"] = config
    _state["max_tokens"] = config["generation"].get("max_tokens", 50)
    _state["specialist_url"] = config["network"]["party_b"]["server_url"]

    from src.model_catalog import build_runtime_config, get_active_ids

    from src.model_catalog import resolve_pair_paths

    print("[practitioner_api] Loading practitioner components (this may take a minute) ...")
    _state["loading"] = True
    try:
        _state["ctx"] = await asyncio.to_thread(_load_practitioner_sync, config)
        pr_id, sp_id = get_active_ids(config)
        runtime = build_runtime_config(config, pr_id, sp_id)
        _state["metrics"] = _load_metrics(runtime)

        pair = resolve_pair_paths(config, pr_id, sp_id, PROJECT_ROOT)
        head_path = PROJECT_ROOT / pair["pair_dir"] / "routing_head.npz"
        if head_path.exists():
            from src.helix import load_routing_head

            weights, bias = load_routing_head(head_path)
            _state["helix_routing_head"] = {"weights": weights, "bias": bias}
            print(f"[practitioner_api] HELIX routing head loaded ({weights.shape})")
        else:
            print(f"[practitioner_api] HELIX: no routing head at {head_path}")
    except Exception as exc:
        print(f"[practitioner_api] Startup failed: {exc}")
        raise
    finally:
        _state["loading"] = False

    await asyncio.to_thread(_hospital_clear_helix)
    print("[practitioner_api] Ready — waiting for passphrase.")
    yield
    _clear_helix_session()
    _unload_practitioner()


app = FastAPI(title="CrossMind Practitioner API (Party B)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigUpdate(BaseModel):
    max_tokens: int | None = None
    specialist_url: str | None = None


class SessionKeyRequest(BaseModel):
    passphrase: str


class HelixKeyRequest(BaseModel):
    helix_key: str
    sync_hospital: bool | None = None


class HelixClassifyRequest(BaseModel):
    prompt: str


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "role": "practitioner",
        "ready": _state["ctx"] is not None and not _state["loading"],
        "loading": _state["loading"],
        "key_set": _state["key_fingerprint"] is not None,
        "key_fingerprint": _state["key_fingerprint"],
        "specialist_url": _state["specialist_url"],
        "generating": _state["generating"],
    }


@app.get("/api/config")
async def get_config():
    return _config_payload()


@app.get("/api/helix/status")
async def helix_status():
    active = _state["helix_ctx"] is not None
    return {
        "active": active,
        "fingerprint": _state["helix_key_fingerprint"] if active else None,
    }


@app.post("/api/config")
async def update_config(body: ConfigUpdate):
    if body.max_tokens is not None:
        if not 1 <= body.max_tokens <= 200:
            raise HTTPException(status_code=400, detail="max_tokens must be 1–200")
        _state["max_tokens"] = body.max_tokens
    if body.specialist_url is not None:
        _state["specialist_url"] = body.specialist_url.strip()
        ctx: TwoPartyContext | None = _state["ctx"]
        if ctx is not None:
            ctx.server_url = _state["specialist_url"]
    return _config_payload()


@app.post("/api/session/key")
async def session_key(body: SessionKeyRequest):
    passphrase = body.passphrase.strip()
    if not passphrase:
        raise HTTPException(status_code=400, detail="Empty passphrase")
    return _set_passphrase(passphrase)


@app.get("/api/phrases")
async def get_phrases():
    return {"phrases": MEDICAL_PHRASES}


@app.post("/api/session/helix-key")
async def session_helix_key(body: HelixKeyRequest):
    if _state["helix_routing_head"] is None:
        raise HTTPException(status_code=400, detail="HELIX routing head not available for this pair")

    helix_key = body.helix_key.strip()
    if not helix_key:
        raise HTTPException(status_code=400, detail="Empty HELIX key")

    from src.helix import create_context, helix_key_fingerprint

    old_fp = _state["helix_key_fingerprint"]
    ctx = create_context()
    fp = helix_key_fingerprint(helix_key)

    _state["helix_ctx"] = ctx
    _state["helix_key_fingerprint"] = fp

    remote = await asyncio.to_thread(_fetch_hospital_helix_status)
    hospital_bootstrapped = bool(remote and remote.get("bootstrapped"))
    key_changed = old_fp is not None and old_fp != fp

    if body.sync_hospital is True:
        should_bootstrap = True
    elif body.sync_hospital is False:
        should_bootstrap = False
    else:
        # First key or same key re-applied: share with hospital. Key rotation: clinic only.
        should_bootstrap = not key_changed or not hospital_bootstrapped

    if should_bootstrap:
        try:
            await asyncio.to_thread(_hospital_bootstrap_helix, ctx, fp)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"HELIX bootstrap failed: {exc}") from exc

    sync = _helix_sync_status()
    action = "set+sync" if should_bootstrap else "set (clinic only)"
    print(f"[practitioner_api] HELIX key {action}: fingerprint={fp} synced={sync['synced']}")
    return {
        "status": "ok",
        "fingerprint": fp,
        "hospital_synced": sync["synced"],
        "hospital_fingerprint": sync["hospital_fingerprint"],
    }


@app.post("/api/session/helix-key/sync")
async def sync_helix_key_to_hospital():
    """Push the current clinic HELIX public context to the hospital."""
    ctx = _state["helix_ctx"]
    fp = _state["helix_key_fingerprint"]
    if ctx is None or fp is None:
        raise HTTPException(status_code=400, detail="Set HELIX key on clinic first")

    try:
        await asyncio.to_thread(_hospital_bootstrap_helix, ctx, fp)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"HELIX bootstrap failed: {exc}") from exc

    sync = _helix_sync_status()
    print(f"[practitioner_api] HELIX hospital sync: fingerprint={fp} synced={sync['synced']}")
    return {
        "status": "ok",
        "fingerprint": fp,
        "hospital_synced": sync["synced"],
        "hospital_fingerprint": sync["hospital_fingerprint"],
    }


@app.post("/api/session/helix-key/clear")
async def clear_helix_key():
    """End HELIX session locally and on hospital (e.g. switch back to Sealed mode)."""
    await asyncio.to_thread(_clear_helix_session)
    return _config_payload()


@app.post("/api/helix/classify")
async def helix_classify(body: HelixClassifyRequest):
    ctx_tp: TwoPartyContext | None = _state["ctx"]
    helix_ctx = _state["helix_ctx"]
    head = _state["helix_routing_head"]

    if ctx_tp is None or _state["loading"]:
        raise HTTPException(status_code=503, detail="Practitioner not ready")
    if head is None:
        raise HTTPException(status_code=400, detail="HELIX routing head not available")
    if helix_ctx is None:
        raise HTTPException(status_code=400, detail="Set HELIX key first (POST /api/session/helix-key)")
    if _state["generating"]:
        raise HTTPException(status_code=409, detail="Sealed generation in progress")
    if _state["helix_classifying"]:
        raise HTTPException(status_code=409, detail="HELIX classification in progress")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    def _run() -> dict:
        import httpx
        import numpy as np
        import torch

        from src.helix import CATEGORIES, split_decrypt, split_encrypt

        sync = _helix_sync_status()
        hospital_synced = sync["synced"]

        model = ctx_tp.qwen_model
        tokenizer = ctx_tp.qwen_tokenizer
        W_star = ctx_tp.W_star.cpu().numpy()
        b_star = ctx_tp.b_star.cpu().numpy()

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(model.device)
        with torch.no_grad():
            output = model.model(input_ids)
            h_B = output.last_hidden_state[0, -1, :].cpu().numpy().astype(np.float32)

        h_aligned = h_B @ W_star + b_star

        ciphertext_b64, sample, size_bytes, encrypt_ms = split_encrypt(helix_ctx, h_aligned)

        def _failure(
            *,
            reason: str,
            message: str,
            compute_ms: float = 0.0,
            decrypt_ms: float = 0.0,
        ) -> dict:
            return {
                "crypto_verified": False,
                "hospital_synced": hospital_synced,
                "failure_reason": reason,
                "message": message,
                "label": None,
                "label_name": None,
                "confidences": None,
                "plaintext_matches": False,
                "ciphertext_sample": sample,
                "ciphertext_size_bytes": size_bytes,
                "timing": {
                    "encrypt_ms": encrypt_ms,
                    "compute_ms": compute_ms,
                    "decrypt_ms": decrypt_ms,
                    "total_ms": encrypt_ms + compute_ms + decrypt_ms,
                },
            }

        if not hospital_synced:
            clinic_fp = sync["clinic_fingerprint"] or "?"
            hospital_fp = sync["hospital_fingerprint"] or "none"
            return _failure(
                reason="hospital_key_mismatch",
                message=(
                    "HELIX key not shared with the hospital. "
                    f"Clinic key {clinic_fp} — hospital still on {hospital_fp}. "
                    "Sync the key before routing."
                ),
            )

        specialist_url = _state["specialist_url"]
        try:
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    f"{specialist_url.rstrip('/')}/api/helix/compute",
                    json={"ciphertext_b64": ciphertext_b64},
                )
                resp.raise_for_status()
                compute_payload = resp.json()
        except Exception as exc:
            return _failure(
                reason="compute_failed",
                message=f"Hospital could not compute on this ciphertext: {exc}",
            )

        logits_b64 = compute_payload["logits_b64"]
        compute_ms = float(compute_payload["compute_ms"])

        try:
            label, confidences, decrypt_ms, plaintext_matches = split_decrypt(
                helix_ctx, logits_b64, h_aligned, head["weights"], head["bias"],
            )
        except Exception as exc:
            return _failure(
                reason="decrypt_failed",
                message=f"Clinic could not decrypt the routing result: {exc}",
                compute_ms=compute_ms,
            )

        crypto_verified = plaintext_matches
        total_ms = encrypt_ms + compute_ms + decrypt_ms
        result = {
            "crypto_verified": crypto_verified,
            "hospital_synced": True,
            "failure_reason": None if crypto_verified else "decrypt_integrity",
            "message": None
            if crypto_verified
            else "Decrypted routing does not match the expected result — HELIX key or session may be invalid.",
            "label": label,
            "label_name": CATEGORIES[label],
            "confidences": dict(zip(CATEGORIES, confidences)),
            "plaintext_matches": plaintext_matches,
            "ciphertext_sample": sample,
            "ciphertext_size_bytes": size_bytes,
            "timing": {
                "encrypt_ms": encrypt_ms,
                "compute_ms": compute_ms,
                "decrypt_ms": decrypt_ms,
                "total_ms": total_ms,
            },
        }
        return result

    _state["helix_classifying"] = True
    try:
        return await asyncio.to_thread(_run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _state["helix_classifying"] = False


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await websocket.accept()
    ctx: TwoPartyContext | None = _state["ctx"]

    if ctx is None or _state["loading"]:
        await websocket.send_json({"type": "error", "message": "Practitioner not ready"})
        await websocket.close()
        return

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            prompt = payload.get("prompt", "").strip()

            if not prompt:
                await websocket.send_json({"type": "error", "message": "Empty prompt"})
                continue

            if _state["generating"]:
                await websocket.send_json({"type": "error", "message": "Generation in progress"})
                continue

            if _state["helix_classifying"]:
                await websocket.send_json({"type": "error", "message": "HELIX classification in progress"})
                continue

            if ctx.R_torch is None:
                await websocket.send_json({
                    "type": "error",
                    "message": "Set passphrase first (POST /api/session/key)",
                })
                continue

            _state["generating"] = True
            max_tokens = payload.get("max_tokens") or _state["max_tokens"]

            try:
                events = await asyncio.to_thread(
                    lambda: list(
                        generate_two_party_stream(ctx, prompt, max_tokens=max_tokens)
                    )
                )
                for event in events:
                    await websocket.send_json(event)
                    await asyncio.sleep(0)
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": str(exc)})
            finally:
                _state["generating"] = False

    except WebSocketDisconnect:
        pass


def main():
    import argparse
    import uvicorn

    config = _load_config()
    net = config["network"].get("party_b_api", {})
    host = net.get("host", "0.0.0.0")
    port = net.get("port", 8421)

    parser = argparse.ArgumentParser(description="CrossMind practitioner API (Party B)")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
