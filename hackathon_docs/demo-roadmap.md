# Demo roadmap

Two modes:

- **Stage (AI BEAVERS):** **3 minutes** — use [pitch.md](pitch.md), not this file.
- **Lab / booth:** **~8 minutes** below — split UI, Sealed + HELIX deep dive.

Read [overview.md](overview.md) first if you are new to the project.

---

## Before you go on stage

- [ ] Stack built and smoke-tested — [BUILD.md](../BUILD.md) § Run the demo
- [ ] `alignment_map.npz`, `lm_head.npy`, `routing_head.npz` present for active pair
- [ ] 36 GB+ unified memory free for float16 7B+8B
- [ ] `./demo_split_all.sh` or four terminals ready
- [ ] Passphrase `hackathon2026` on both UIs; HELIX key on **clinic only** (e.g. `helix-demo-2026`)
- [ ] Backup screen recording of Sealed + HELIX run

**Quick start:** `./demo_split_all.sh` → clinic http://localhost:4200, hospital http://localhost:4201

---

## What to show

| Story | Show | Status |
| ----- | ---- | ------ |
| Architectural split | Clinic has prompt; hospital wire tab has packets only | Working |
| Cross-model quality | Chest-pain prompt → coherent continuation | Working (Sealed) |
| Sealed obfuscation | Wrong hospital passphrase → garbage tokens | Working |
| HELIX routing | Clinic routing card; hospital heatmap **without** department name | Working |
| Honest limits | Verbal: server decrypts for gen; ~83% routing; research demo | Required |

---

## Recommended flow (8 min)

| Time | Block |
| ---- | ----- |
| 0:00 | Problem — want best model without shipping raw chart text |
| 1:00 | Privacy ladder steps 1–3 (overview table) |
| 2:00 | **Live** Sealed generation — split UI, tile clinic + hospital |
| 4:00 | **Live** wrong passphrase on hospital |
| 5:00 | Sealed ≠ HE; server decrypts every generation step |
| 6:00 | **Live** HELIX mode — routing card vs hospital compute tab |
| 7:00 | Limits + DEMO disclaimer |
| 8:00 | Close — “path, not product” |

---

## Spoken script

*Do not say “HIPAA compliant”, “end-to-end encrypted LLM”, or “hospital never sees patient data” without qualifiers.*

### Hook (0:00)

> “Hospitals want the best AI model, but they shouldn’t ship every patient note to someone else’s GPU as plain text. We split the job: a **clinic model** reads the chart locally; a **specialist model** works on **vectors**, not your sentence. Sealed scrambles those vectors on the wire; HELIX adds real cryptography where the math fits — **five-class routing**, not full chat.”

### Ladder (1:00)

> “Step one: **Qwen stays on the clinic** — the hospital never gets the prompt string. Step two: alignment — the hospital still receives numbers that encode meaning. Step three: **Sealed** — rotation on the wire; the hospital **reverses it every token** to generate. That blocks a **network sniffer**, not a curious hospital server.”

### Sealed live (2:00)

1. Both UIs: `hackathon2026` → Set key  
2. Clinic: **Sealed**, chest-pain prompt  
3. Point at hospital wire panel  

> “The hospital **doesn’t receive the prompt string**; it **decrypts rotated vectors** each step. Cross-vendor: Qwen encodes, Llama decodes.”

CLI fallback: `uv run python scripts/demo_e2e.py --max-tokens 40 --passphrase hackathon2026`

### Wrong key (4:00)

Hospital passphrase → `wrong` → Set key. Clinic sends again.

> “Same bytes on the wire — wrong secret → garbage. **Eavesdropper** story, not anti-hospital story.”

### HELIX (6:00)

Clinic: **HELIX**, set HELIX key, routing-style prompt.

> “Routing is five logits, not 128k. Hospital does blind CKKS matmul — no plaintext vector, no department name. Clinic decrypts. ~3–4 seconds; ~83% val accuracy on held-out labels.”

CLI fallback: `uv run python scripts/demo_helix.py`

### Close (8:00)

> “**Today:** split inference, Sealed obfuscation, HELIX routing crypto. **Not today:** full encrypted chat, clinical deployment, compliance certification. Questions?”

---

## HELIX categories

| ID | Department | Example cues |
|----|------------|--------------|
| 0 | Cardiology | chest pain, troponin, ECG |
| 1 | Neurology | stroke, seizure, headache |
| 2 | Oncology | tumor, biopsy, chemo |
| 3 | Orthopedics | fracture, joint, back pain |
| 4 | General_Medicine | fever, diabetes, general symptoms |

---

## Banned phrases

| ❌ Too strong | ✅ Say instead |
| ------------- | -------------- |
| Hospital never sees patient data | Hospital doesn’t get the **prompt string**; vectors still encode information |
| Encrypted vectors (Sealed) | **Rotated** vectors — shared-secret obfuscation |
| End-to-end encrypted LLM | Sealed for generation; HELIX for routing only |
| HIPAA compliant | Minimization + selective HE — **not certified compliance** |

---

## Fallbacks

| Failure | Fallback |
| ------- | -------- |
| Split UI | `demo_e2e.py` + `demo_helix.py` |
| OOM | Lower `max_tokens`; stop `label_routing.py` |
| HELIX slow | Narrate latency; use recording |
| Missing artifacts | [BUILD.md](../BUILD.md) — train alignment first |
