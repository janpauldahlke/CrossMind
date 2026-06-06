# Build guide — from zero to running demo

Step-by-step instructions to install dependencies, download models, train alignment, and run the split clinic/hospital demo.

For **what CrossMind is and why**, read [hackathon_docs/overview.md](hackathon_docs/overview.md).

---

## Requirements

### Hardware

| Platform  | GPU               | Memory         | Notes                                                                                        |
| --------- | ----------------- | -------------- | -------------------------------------------------------------------------------------------- |
| **macOS** | Apple Silicon MPS | 36 GB+ unified | float16 Qwen 7B + Llama 8B; avoid running both backends + heavy training jobs simultaneously |

### Software

| Tool                                      | Purpose                               |
| ----------------------------------------- | ------------------------------------- |
| **[uv](https://github.com/astral-sh/uv)** | Python env and `uv run` (recommended) |
| **Python 3.11+**                          | Backend                               |
| **Node.js 18+** and **npm**               | Angular clinic/hospital frontends     |
| **Git**                                   | Clone repo                            |
| **Hugging Face account**                  | Download gated models (Llama)         |

### Python stack (via `requirements.txt`)

PyTorch, Hugging Face `transformers`, `accelerate`, `numpy`, `scipy`, `datasets`, `fastapi`, `uvicorn`, `tenseal` (HELIX), `pytest`.

**Inference engine:** PyTorch + Hugging Face — **not** the llama.cpp binary. The `config.yaml` points at a local folder named `llama.cpp/models/`; that is just where **Hugging Face-format** weight directories live on disk.

---

## Step 0 — Clone and install

```bash
git clone <repo-url> CrossMind
cd CrossMind

uv venv
uv pip install -r requirements.txt

cd frontend && npm install && cd ..
```

Edit `config.yaml` model paths to point at your local Hugging Face weight directories. Demo weights live under `data/pairs/`; **`alignment_map.npz`** and **`routing_head.npz`** are in git; **`lm_head.npy`** (~2 GB) **is not**!!
Please run `train_alignment.py` once after clone or copy from your local training run (see [data/pairs/README.md](data/pairs/README.md)).

---

## Step 1 — Download models

Default pair: **Qwen 2.5 7B Instruct** (clinic) + **Llama 3.1 8B Instruct** (hospital).

1. Install the Hugging Face CLI: `uv pip install huggingface_hub`
2. Log in: `huggingface-cli login` (Llama weights are gated)
3. Download into a directory you will reference in config, e.g. `~/models/`:

```bash
mkdir -p ~/models

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ~/models/Qwen2.5-7B-Instruct

huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/models/Meta-Llama-3.1-8B-Instruct
```

4. Edit **`config.yaml`** — set `models.qwen.model_id`, `models.llama.model_id`, and matching `model_catalog.*.model_path` entries to your local paths.

---

## Step 2 — Smoke test (gate)

Verifies MPS, model load, hidden-state extraction, and LM-head split:

```bash
uv run python scripts/smoke_test.py
```

Fix paths or memory issues before continuing. Expect a few minutes on first model download/cache.

---

## Step 3 — Prepare alignment dataset

Downloads public text (Alpaca, PubMedQA, ChatDoctor) into `data/public_dataset/texts.jsonl`:

```bash
uv run python scripts/prepare_data.py
```

Skip if `data/public_dataset/texts.jsonl` already exists (~4000 samples).

---

## Step 4 — Extract hidden states (long)

Three passes in one script: specialist states -> practitioner states -> token-aligned pairs.

```bash
uv run python scripts/extract_states.py \
  --practitioner-id qwen-2.5-7b \
  --specialist-id llama-3.1-8b
```

**Outputs:**

```
data/raw/{pair}/specialist/*.npz      # Llama hidden states per sample
data/raw/{pair}/practitioner/*.npz    # Qwen hidden states per sample
data/aligned/{pair}/chunk_*.npz       # aligned Z_A, Z_B pairs for training
```

**Time:** hours on MPS (4000 samples x two models). Resumes automatically — re-run the same command after interrupt.

**Memory:** One full model loaded at a time during extraction.

---

## Step 5 — Train alignment

Learns `W*`, `b*` via ridge regression; saves map + metrics; extracts specialist LM head if missing:

```bash
uv run python scripts/train_alignment.py \
  --practitioner-id qwen-2.5-7b \
  --specialist-id llama-3.1-8b
```

**Outputs:**

```
data/pairs/qwen-2.5-7b+llama-3.1-8b/alignment_map.npz
data/pairs/qwen-2.5-7b+llama-3.1-8b/alignment_metrics.json
data/pairs/qwen-2.5-7b+llama-3.1-8b/lm_head.npy
data/pairs/qwen-2.5-7b+llama-3.1-8b/tokenizer/
```

Target: cosine similarity **>= ~0.5** (demo pair typically ~0.83). Check `alignment_metrics.json`.

---

## Step 6 — Verify generation (single machine)

```bash
uv run python scripts/demo_generation.py
```

Runs medical test prompts through cross-model decoding. Optional sanity check before networking.

Other quick checks:

```bash
uv run python scripts/demo_encrypted.py    # Sealed correct vs wrong key
uv run python scripts/demo_e2e.py --max-tokens 40 --passphrase hackathon2026
```

---

## Step 7 — HELIX routing (optional)

Requires specialist raw states from Step 4 and labels.

### 7a — Routing labels (resumable)

```bash
PYTHONUNBUFFERED=1 uv run python scripts/label_routing.py
# resume: uv run python scripts/label_routing.py --resume
```

Writes `data/routing_labels.jsonl` (medical samples -> 5 departments).

### 7b — Train routing head

```bash
uv run python scripts/train_routing.py \
  --practitioner-id qwen-2.5-7b \
  --specialist-id llama-3.1-8b
```

Writes `data/pairs/qwen-2.5-7b+llama-3.1-8b/routing_head.npz` (~83% val accuracy).

### 7c — CLI HELIX demo

```bash
uv run python scripts/demo_helix.py
```

---

## Step 8 — Run the split demo (hackathon UI)

**All four processes:**

```bash
./demo_split_all.sh
```

| Service      | URL                   |
| ------------ | --------------------- |
| Clinic UI    | http://localhost:4200 |
| Hospital UI  | http://localhost:4201 |
| Clinic API   | http://localhost:8421 |
| Hospital API | http://localhost:8420 |

**Manual (easier when debugging):**

```bash
# Terminal 1 — hospital / specialist
uv run python -m src.server --port 8420

# Terminal 2 — clinic / practitioner
uv run python -m src.practitioner_api --port 8421

# Terminal 3
cd frontend && npm run start:clinic

# Terminal 4
cd frontend && npm run start:hospital
```

**Demo controls:**

- Passphrase **`hackathon2026`** on **both** UIs -> Set key (Sealed)
- Clinic mode **Sealed** — cross-model generation
- Clinic mode **HELIX** — set HELIX key on clinic only (e.g. `helix-demo-2026`)
- Wrong-key demo: different passphrase on **hospital** only

Presentation script: [hackathon_docs/demo-roadmap.md](hackathon_docs/demo-roadmap.md).

**Backends only:**

```bash
bash scripts/demo_split.sh
```

---

## Pipeline summary

```
config.yaml paths
    |
smoke_test.py
    |
prepare_data.py          -> data/public_dataset/texts.jsonl
    |
extract_states.py        -> data/raw/.../chunk_*.npz
    |
train_alignment.py       -> alignment_map.npz, lm_head.npy, tokenizer/
    |
demo_generation.py       (sanity)
    |
[optional] label_routing.py -> routing_labels.jsonl
[optional] train_routing.py -> routing_head.npz
    |
demo_split_all.sh        (clinic + hospital UI)
```

---

## Troubleshooting

| Symptom                    | Check                                                     |
| -------------------------- | --------------------------------------------------------- |
| `Dataset not found`        | Run `prepare_data.py`                                     |
| `alignment_map not found`  | Run `train_alignment.py` with matching pair ids           |
| `HELIX disabled in UI`     | Missing `routing_head.npz` — Step 7                       |
| Garbage tokens on hospital | Passphrase mismatch — both sides must share Sealed key    |
| OOM on Mac                 | One backend at a time; avoid simultaneous training + demo |

Run tests: `uv run pytest`

---

## Artifact checklist (active pair)

Before demo, confirm:

- [ ] `data/pairs/qwen-2.5-7b+llama-3.1-8b/alignment_map.npz`
- [ ] `data/pairs/qwen-2.5-7b+llama-3.1-8b/lm_head.npy`
- [ ] `data/pairs/qwen-2.5-7b+llama-3.1-8b/tokenizer/`
- [ ] `data/pairs/qwen-2.5-7b+llama-3.1-8b/routing_head.npz` (HELIX)
