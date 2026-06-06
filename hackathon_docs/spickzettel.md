# Spickzettel — how CrossMind works (technical)

Plain language, no analogies. For the privacy story see [overview.md](overview.md).

---

## What runs where

| Process | Port | Machine role | Loads |
| ------- | ---- | -------------- | ----- |
| **Clinic UI** | 4200 | Browser | Angular frontend |
| **Hospital UI** | 4201 | Browser | Angular frontend |
| **`practitioner_api`** | 8421 | **Clinic backend** | Full **Qwen 2.5 7B**, alignment map `W*`/`b*`, rotation key `R`, HELIX secret key |
| **`server`** | 8420 | **Hospital backend** | **Llama LM head** only (weight matrix + bias), rotation key `R`, HELIX **public** context (no secret key) |

The **user prompt text** is typed in the clinic UI. It never appears in HTTP payloads to the hospital as a string.

---

## One autoregressive step (Sealed / generation mode)

1. **Clinic — forward pass**  
   Qwen reads the current token sequence and outputs a hidden vector `h_B` (last token position, dimension 3584).

2. **Clinic — linear alignment**  
   `h_aligned = h_B @ W* + b*`  
   Produces a vector in Llama’s hidden space (dimension 4096).  
   `W*` and `b*` were learned offline from paired token examples (ridge regression).

3. **Clinic — Sealed (rotation)**  
   `h_enc = h_aligned @ R`  
   `R` is an orthogonal matrix derived from the shared passphrase. Same passphrase on both sides.

4. **Network**  
   JSON/array of 4096 floats sent to hospital `POST /infer`.

5. **Hospital — decrypt**  
   `h_dec = h_enc @ Rᵀ`  
   Recovers `h_aligned` (lossless if keys match).

6. **Hospital — LM head**  
   `logits = h_dec @ W_lmᵀ + b_lm`  
   `W_lm` is Llama’s vocabulary projection (shape ≈ 128256 × 4096), saved as `lm_head.npy`.  
   Hospital does **not** load the full Llama transformer — only this head.

7. **Hospital → clinic**  
   Returns one token ID (or logits). Clinic decodes with Llama tokenizer, feeds next tokens back into Qwen. Repeat until stop or max tokens.

**Important:** the hospital **must decrypt** each step to run the LM head. Sealed protects a **wire eavesdropper**, not the hospital server.

---

## LM head (what it is)

The **LM head** is the final linear layer of a causal LM: hidden state → scores over the vocabulary.  
Cross-model generation reuses **Llama’s** head on **aligned Qwen-derived** hidden states.  
That only works if alignment puts `h_aligned` where Llama’s head expects it.

---

## Linear alignment (what it is)

Two models assign a vector to each token, but in different coordinate systems.  
**Linear alignment** learns a single affine map:

- `W*` — matrix (3584 → 4096)  
- `b*` — bias (4096)

Trained so aligned Qwen states match Llama states on the same text positions (public medical/general corpus).  
Reported quality in this repo: cosine similarity ≈ **0.83** on validation pairs.

Alignment **maps** meaning into Llama space; it does **not** hide content from whoever receives `h_aligned`.

---

## Sealed vs HELIX

| | **Sealed (rotation)** | **HELIX (CKKS)** |
|---|----------------------|------------------|
| **Used for** | Token-by-token **generation** | One-shot **5-class routing** |
| **Operation** | Multiply vector by shared orthogonal `R` | Encrypt vector with CKKS; hospital multiplies ciphertext by routing matrix |
| **Hospital sees** | Decrypted `h_aligned` every step | Ciphertext in; encrypted logits out — **no plaintext vector, no label** |
| **Speed** | Fast (per token) | ~3–4 s CPU per request |
| **Crypto type** | Shared secret (obfuscation) | Homomorphic encryption (TenSEAL / CKKS) |

---

## HELIX routing (one request)

1. Clinic: Qwen forward → `h_B` → align → `h_aligned` (4096 floats).

2. Clinic: **CKKS encrypt** `h_aligned` with clinic-held **secret key**; send ciphertext + public context to hospital.

3. Hospital: homomorphic multiply  
   `encrypted_logits = Enc(h_aligned) @ W_route + b_route`  
   `W_route` is 4096 × 5 (Cardiology, Neurology, Oncology, Orthopedics, General Medicine).  
   Hospital never decrypts.

4. Clinic: **decrypt** 5 logits → argmax → department name + confidence bars in UI.

5. Hospital UI: timing + ciphertext heatmap only — **no department name**.

Routing head val accuracy ≈ **83%**; demo prompts are chosen to be clear.

---

## CKKS (what it is)

**CKKS** is a homomorphic encryption scheme for approximate real numbers.  
It allows:

- encrypt vector `x` → ciphertext `Enc(x)`  
- compute `Enc(x) @ W + b` on the server **without** decrypting `x`  
- only the key holder decrypts the result

**TenSEAL** wraps Microsoft SEAL in Python.  
Tradeoff: practical for **small output dimension** (5 classes), not for full vocabulary (128k logits).

---

## Keys and passphrases (demo)

| Key | Where set | Purpose |
| --- | --------- | ------- |
| **Shared passphrase** | Both UIs → `hackathon2026` | Derives rotation matrix `R` (Sealed) |
| **HELIX key** | **Clinic only** | CKKS key pair; hospital gets public context fingerprint |

Wrong hospital passphrase → same wire bytes, garbage tokens (wrong-key demo).

---

## Files on disk

```
data/pairs/qwen-2.5-7b+llama-3.1-8b/
  alignment_map.npz   <- W*, b*        (in git)
  routing_head.npz    <- HELIX weights (in git)
  lm_head.npy         <- Llama head     (~2 GB, local only)
  tokenizer/          <- Llama tokenizer (in git)
```

---

## What each side learns (one glance)

| | Prompt text | Aligned vector | Routing label |
|---|-------------|----------------|---------------|
| **Clinic** | Yes | Yes (plaintext) | Yes (after HELIX decrypt) |
| **Hospital (Sealed gen)** | No | Yes (after Rᵀ) | — |
| **Hospital (HELIX)** | No | No (ciphertext only) | No |
| **Wire eavesdropper (Sealed)** | No | Sees rotated floats, not usable without R | — |

---

## Commands (demo day)

```bash
./demo_split_all.sh
# Clinic  http://localhost:4200   Hospital  http://localhost:4201
# Sealed: passphrase both sides
# HELIX:  HELIX key on clinic only
```

---

## Banned one-liners (if someone asks)

- Not “hospital never sees patient data” → hospital doesn’t get **prompt string**; vectors still encode information; Sealed server **decrypts for generation**.
- Not “HIPAA compliant” → minimization + selective HE; **not certified**.
- Not “fully encrypted LLM” → **HELIX = routing only**; generation uses Sealed rotation.
