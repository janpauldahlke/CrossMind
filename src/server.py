"""
Party A server: lightweight FastAPI service for privacy-preserving inference.

Receives rotation-encrypted hidden-state vectors from Party B (Qwen client),
decrypts them with the shared rotation key R, applies Llama's LM head, and
returns the decoded token.  The entire inference pipeline is pure numpy --
no ML framework required on the server side.

Wire protocol uses base64-encoded float32 byte buffers for compactness
(~22 KB per 4096-dim vector vs ~80 KB as a JSON float array).
"""

import argparse
import asyncio
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.encryption import derive_rotation_key, key_fingerprint, verify_key

VECTOR_SAMPLE_SIZE = 64


# ---------------------------------------------------------------------------
# Global state (populated at startup / session key)
# ---------------------------------------------------------------------------

_state: dict = {
    "lm_head": None,       # (vocab_size, hidden_dim) float32
    "tokenizer": None,     # HuggingFace tokenizer (loaded once)
    "rotation_key": None,  # (hidden_dim, hidden_dim) float32
    "hidden_dim": None,
    "last_infer": None,
    "last_helix": None,
    "activity_subscribers": set(),
    "models": None,
    "helix_routing_head": None,
    "helix_pub_ctx": None,
    "helix_key_fingerprint": None,
    "practitioner_url": None,
}


def _load_config() -> dict:
    from src.models import load_config

    return load_config()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clear_helix_state() -> None:
    _state["helix_pub_ctx"] = None
    _state["helix_key_fingerprint"] = None


async def _sync_helix_from_practitioner() -> None:
    """Mirror clinic HELIX session — practitioner is source of truth."""
    url = _state["practitioner_url"]
    if not url:
        return
    try:
        import httpx

        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/api/helix/status")
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        return

    if not data.get("active"):
        if _state["helix_pub_ctx"] is not None:
            _clear_helix_state()
            print("[server] HELIX cleared — clinic session inactive")
        return

    fp = data.get("fingerprint")
    if fp and _state["helix_key_fingerprint"] and fp != _state["helix_key_fingerprint"]:
        _clear_helix_state()
        print(f"[server] HELIX cleared — fingerprint mismatch (clinic={fp})")


async def _broadcast_activity(event: dict) -> None:
    dead: list[WebSocket] = []
    for ws in list(_state["activity_subscribers"]):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _state["activity_subscribers"].discard(ws)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.model_catalog import build_runtime_config, get_active_ids

    from src.model_catalog import catalog_response, default_catalog_from_legacy

    config = _load_config()
    if "model_catalog" not in config:
        config["model_catalog"] = default_catalog_from_legacy(config)
    pr_id, sp_id = get_active_ids(config)
    config = build_runtime_config(config, pr_id, sp_id)
    _state["models"] = catalog_response(config, {}, PROJECT_ROOT)
    gen_cfg = config["generation"]

    lm_head_path = gen_cfg["lm_head_path"]
    print(f"[server] Loading LM head from {lm_head_path} ...")
    _state["lm_head"] = np.load(lm_head_path).astype(np.float32)
    print(f"[server]   shape = {_state['lm_head'].shape}")
    _state["hidden_dim"] = _state["lm_head"].shape[1]

    tok_path = gen_cfg["tokenizer_path"]
    print(f"[server] Loading tokenizer from {tok_path} ...")
    from transformers import AutoTokenizer
    _state["tokenizer"] = AutoTokenizer.from_pretrained(
        tok_path, clean_up_tokenization_spaces=False,
    )
    from src.model_catalog import resolve_pair_paths

    pair = resolve_pair_paths(config, pr_id, sp_id, PROJECT_ROOT)
    head_path = PROJECT_ROOT / pair["pair_dir"] / "routing_head.npz"

    net = config["network"]
    api_port = net.get("party_b_api", {}).get("port", 8421)
    _state["practitioner_url"] = f"http://localhost:{api_port}"

    if head_path.exists():
        from src.helix import load_routing_head

        weights, bias = load_routing_head(head_path)
        _state["helix_routing_head"] = {"weights": weights, "bias": bias}
        print(f"[server] HELIX routing head loaded ({weights.shape})")
    else:
        print(f"[server] HELIX: no routing head at {head_path}")

    print("[server] Ready — waiting for session passphrase.")

    yield

    _state["lm_head"] = None
    _state["tokenizer"] = None
    _state["rotation_key"] = None
    _state["last_infer"] = None
    _state["last_helix"] = None
    _state["activity_subscribers"].clear()
    _state["models"] = None
    _state["helix_routing_head"] = None
    _state["helix_pub_ctx"] = None
    _state["helix_key_fingerprint"] = None
    _state["practitioner_url"] = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="CrossMind Specialist Server (Party A)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class HandshakeRequest(BaseModel):
    key_b64: str
    dim: int


class SessionKeyRequest(BaseModel):
    passphrase: str


class InferRequest(BaseModel):
    vector_b64: str
    step: int = 0


class InferResponse(BaseModel):
    token_id: int
    token_text: str


class HelixBootstrapRequest(BaseModel):
    public_ctx_b64: str
    fingerprint: str


class HelixComputeRequest(BaseModel):
    ciphertext_b64: str


class HelixComputeResponse(BaseModel):
    logits_b64: str
    compute_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config")
async def get_config():
    """Active model pair for the specialist UI."""
    await _sync_helix_from_practitioner()
    models = _state["models"]
    if models is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    bootstrapped = _state["helix_pub_ctx"] is not None
    return {
        "role": "specialist",
        "ready": True,
        "helix_available": _state["helix_routing_head"] is not None,
        "helix_bootstrapped": bootstrapped,
        "helix_key_fingerprint": _state["helix_key_fingerprint"] if bootstrapped else None,
        **models,
    }


@app.get("/api/helix/status")
async def helix_status():
    bootstrapped = _state["helix_pub_ctx"] is not None
    return {
        "bootstrapped": bootstrapped,
        "fingerprint": _state["helix_key_fingerprint"] if bootstrapped else None,
    }


@app.get("/health")
async def health():
    R = _state["rotation_key"]
    return {
        "status": "ok",
        "role": "specialist",
        "key_loaded": R is not None,
        "key_fingerprint": key_fingerprint(R) if R is not None else None,
        "lm_head_shape": list(_state["lm_head"].shape) if _state["lm_head"] is not None else None,
    }


@app.post("/api/session/key")
async def session_key(req: SessionKeyRequest):
    """Derive rotation key locally from shared passphrase (no key on wire)."""
    dim = _state["hidden_dim"]
    if dim is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    passphrase = req.passphrase.strip()
    if not passphrase:
        raise HTTPException(status_code=400, detail="Empty passphrase")

    R = derive_rotation_key(dim, passphrase)
    stats = verify_key(R)
    _state["rotation_key"] = R
    fp = key_fingerprint(R)
    print(f"[server] Session key derived: fingerprint={fp} valid={stats['is_valid']}")
    return {
        "status": "ok",
        "fingerprint": fp,
        "orthogonality_ok": stats["is_valid"],
        "max_error": stats["max_error"],
    }


@app.get("/api/last_infer")
async def last_infer():
    if _state["last_infer"] is None:
        return {"received": False}
    return {"received": True, **_state["last_infer"]}


@app.post("/handshake")
async def handshake(req: HandshakeRequest):
    """Legacy: receive full rotation matrix (CLI compatibility)."""
    raw = base64.b64decode(req.key_b64)
    R = np.frombuffer(raw, dtype=np.float32).reshape(req.dim, req.dim).copy()

    identity = np.eye(req.dim, dtype=np.float32)
    max_err = float(np.max(np.abs(R.T @ R - identity)))
    if max_err > 1e-3:
        raise HTTPException(
            status_code=400,
            detail=f"Key failed orthogonality check (max_error={max_err:.2e})",
        )

    _state["rotation_key"] = R
    print(f"[server] Rotation key received: {R.shape}, orthogonality max_err={max_err:.2e}")
    return {"status": "ok", "dim": req.dim, "max_error": max_err}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    R = _state["rotation_key"]
    if R is None:
        raise HTTPException(
            status_code=400,
            detail="No rotation key — set passphrase via POST /api/session/key first",
        )

    lm_head = _state["lm_head"]
    tokenizer = _state["tokenizer"]
    hidden_dim = _state["hidden_dim"]

    raw = base64.b64decode(req.vector_b64)
    h_enc = np.frombuffer(raw, dtype=np.float32).copy()

    if h_enc.shape[0] != hidden_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension mismatch: got {h_enc.shape[0]}, expected {hidden_dim}",
        )

    h_dec = h_enc @ R.T
    logits = h_dec @ lm_head.T
    token_id = int(np.argmax(logits))
    token_text = tokenizer.decode([token_id], skip_special_tokens=False)

    activity = {
        "type": "packet",
        "step": req.step,
        "bytes_on_wire": len(raw),
        "vector_sample": h_enc[:VECTOR_SAMPLE_SIZE].astype(float).tolist(),
        "vector_b64_prefix": req.vector_b64[:48],
        "received_at": _utc_now(),
    }
    _state["last_infer"] = activity
    await _broadcast_activity(activity)

    return InferResponse(token_id=token_id, token_text=token_text)


@app.post("/api/helix/clear")
async def helix_clear():
    """Drop HELIX public context when clinic session ends or mode switches to Sealed."""
    had = _state["helix_pub_ctx"] is not None
    _clear_helix_state()
    if had:
        print("[server] HELIX bootstrap cleared")
    return {"status": "ok", "cleared": had}


@app.post("/api/helix/bootstrap")
async def helix_bootstrap(req: HelixBootstrapRequest):
    """Receive public CKKS context from clinic (no secret key on wire)."""
    if _state["helix_routing_head"] is None:
        raise HTTPException(status_code=400, detail="HELIX routing head not available")

    fingerprint = req.fingerprint.strip()
    if not fingerprint:
        raise HTTPException(status_code=400, detail="Empty HELIX fingerprint")

    from src.helix import context_from_b64

    try:
        _state["helix_pub_ctx"] = context_from_b64(req.public_ctx_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid public context: {exc}") from exc

    _state["helix_key_fingerprint"] = fingerprint
    print(f"[server] HELIX bootstrap: fingerprint={fingerprint}")
    return {"status": "ok", "fingerprint": fingerprint}


@app.post("/api/helix/compute", response_model=HelixComputeResponse)
async def helix_compute(req: HelixComputeRequest):
    """Homomorphic routing on ciphertext — never returns label or confidences."""
    pub_ctx = _state["helix_pub_ctx"]
    head = _state["helix_routing_head"]
    if head is None:
        raise HTTPException(status_code=400, detail="HELIX routing head not available")
    if pub_ctx is None:
        raise HTTPException(
            status_code=400,
            detail="HELIX not bootstrapped — set HELIX key on clinic first",
        )

    from src.helix import extract_ciphertext_sample, split_compute, vector_from_b64

    raw = base64.b64decode(req.ciphertext_b64)
    ct = vector_from_b64(pub_ctx, req.ciphertext_b64)
    input_sample, _ = extract_ciphertext_sample(ct)

    logits_b64, compute_ms, _, _ = split_compute(
        pub_ctx, req.ciphertext_b64, head["weights"], head["bias"],
    )

    activity = {
        "type": "helix_compute",
        "bytes_on_wire": len(raw),
        "ciphertext_sample": input_sample,
        "compute_ms": compute_ms,
        "received_at": _utc_now(),
    }
    _state["last_helix"] = activity
    await _broadcast_activity(activity)

    return HelixComputeResponse(logits_b64=logits_b64, compute_ms=compute_ms)


@app.websocket("/ws/activity")
async def ws_activity(websocket: WebSocket):
    """Broadcast inbound /infer packets to hospital UI (wire visibility)."""
    await websocket.accept()
    _state["activity_subscribers"].add(websocket)

    if _state["last_infer"] is not None:
        await websocket.send_json(_state["last_infer"])
    if _state["last_helix"] is not None:
        await websocket.send_json(_state["last_helix"])

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _state["activity_subscribers"].discard(websocket)


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.server
# ---------------------------------------------------------------------------

def main():
    config = _load_config()
    net = config["network"]["party_a"]
    host = net.get("host", "0.0.0.0")
    port = net.get("port", 8420)

    parser = argparse.ArgumentParser(description="CrossMind specialist server (Party A)")
    parser.add_argument("--host", default=host)
    parser.add_argument("--port", type=int, default=port)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
