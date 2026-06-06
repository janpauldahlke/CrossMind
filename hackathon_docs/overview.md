# CrossMind — overview

**Research demo.** Not medical advice, not a clinical product, not certified compliance.

**One-line pitch:** We split sensitive text across two models: a local encoder and a remote specialist that works on vectors, not your sentence. Sealed obfuscates vectors on the wire; HELIX proves encrypted specialty routing where the math fits. Full encrypted generation is future work.

---

## What problem this explores

Organizations want a **strong central model** but hesitate to ship **raw sensitive text** to that model’s host. CrossMind asks: can a **local model encode** the text and a **remote model decode** from aligned hidden vectors instead?

The idea builds on linear alignment between LLM representation spaces ([Gorbett & Jana, 2025](https://arxiv.org/pdf/2603.18908)): a matrix map `W*` re-expresses one model’s hidden state in another’s basis so the specialist **LM head** can read it.

| Party | Role | Demo model | What it holds |
| ----- | ---- | ---------- | ------------- |
| **Clinic** (Party B) | Encodes user text, aligns, optionally encrypts | Qwen 2.5 7B | Prompt string stays here |
| **Hospital** (Party A) | Applies LM head or HELIX routing head | Llama 3.1 8B | Vectors per request — **not** the prompt string |

The hospital **does not receive the prompt string**. It **does** receive numeric vectors that still encode information. Under **Sealed** generation it **decrypts every step** to run the LM head.

---

## Three layers (do not conflate)

| Layer | Mechanism | Protects *from* | Does *not* protect *from* |
| ----- | --------- | --------------- | --------------------------- |
| **Architectural split** | Vectors cross the network, not prompt text | Hospital disk/logs storing raw chart text | Curious hospital server |
| **Sealed (rotation R)** | `h_enc = h_aligned @ R`; server applies `Rᵀ` | Passive eavesdropper without `R` | Honest-but-curious specialist (holds `Rᵀ`) |
| **HELIX (CKKS)** | Blind matmul on ciphertext routing head | Curious specialist **for routing** | Full-vocab generation; inversion on gen vectors |

---

## Privacy ladder

| Step | What happens | Who learns what | Real privacy gain? |
| ---- | ------------ | --------------- | ------------------ |
| **1. Prompt stays on clinic** | Qwen runs locally | Hospital doesn’t get raw chart text | Data minimization — not encryption alone |
| **2. Alignment** | `h_aligned = h_B @ W* + b*` | Hospital still gets vectors | Maps spaces; doesn’t hide from decryptor |
| **3. Sealed** | Rotated vectors on wire; server reverses R | Eavesdropper sees garbage; **server decrypts for gen** | Wire tap only |
| **4. HELIX** | CKKS blob; blind 5-class matmul | Hospital no plaintext vector or label | Crypto for routing (~3–4 s CPU) |
| **5. Hospital UI** | Packets / heatmap / timing | No prompt, no HELIX department name | Demo visibility — not a security boundary |

---

## Architecture (generation + routing)

```
Clinic (Qwen)                         Network                    Hospital (Llama)
─────────────                         ───────                    ─────────────────
User text (local)
  → h_B
  → h_aligned = h_B @ W* + b*
  → Sealed: h_enc = h_aligned @ R  ── h_enc ──►  h_dec = h_enc @ Rᵀ → LM head → token
                                    ◄─ token ───   (repeat)

HELIX (separate request, routing only):
  → align → CKKS encrypt  ── ciphertext ──►  homomorphic matmul (5 classes)
  ← decrypt label on clinic only
```

---

## Is this actually useful?

| Useful as… | Not useful as… |
| ---------- | -------------- |
| Split inference (local encode, remote decode) | “Privacy-preserving ChatGPT for hospitals” today |
| Wire obfuscation (Sealed) | HIPAA compliance or certified privacy |
| HE where linear ops fit (HELIX, ~83% val routing accuracy) | Full encrypted generation at practical latency |
| A credible research path | A guarantee the hospital learns nothing from vectors |

**Known limits:** routing val accuracy ~**82.7%**; alignment cosine ~**0.83**; generation requires **server-side decrypt** under Sealed; **representation inversion** is open research; UI shows **DEMO — Research prototype only.**

---

## What runs in this repo

| Capability | How |
| ---------- | --- |
| Cross-model generation | Sealed two-party loop (`server` + `practitioner_api` or `demo_e2e.py`) |
| Wrong-key wire demo | Mismatched passphrase on hospital UI |
| HELIX routing | Clinic HELIX mode; `demo_helix.py` |
| Training pipeline | `prepare_data` → `extract_states` → `train_alignment` → optional routing |

To **build from scratch**, see [BUILD.md](../BUILD.md) at the repo root.

For **live presentation**, see [demo-roadmap.md](demo-roadmap.md) (lab walkthrough) or [pitch.md](pitch.md) (3-minute judge pitch).

For **other industries** (pilot sketches), see [use_cases.md](use_cases.md).

---

## Pitch framing (for judges)

CrossMind is strongest when pitched as **infrastructure for a narrow workflow**, not “private ChatGPT for hospitals.”

| Lens | Our answer |
| ---- | ---------- |
| **Buyer** | Hospital-group AI lead / clinic IT blocking raw chart export to a central model |
| **Wedge** | Local encode + remote specialist on **vectors**; **HELIX** = encrypted **5-class routing** only |
| **Why now** | Published linear alignment + CKKS practical for small heads + minimization pressure in regulated buyers |
| **Weakest point** | Generation needs server decrypt (Sealed); ~83% routing; not clinical product or compliance cert |
| **Evidence** | Working split demo + alignment/routing metrics — add one real design-partner signal if you have it |

Full deck outline: [pitch.md](pitch.md).

---

## FAQ

**What does `h_B @ W* + b*` do?**  
Re-expresses Qwen’s hidden vector in Llama’s basis so Llama’s LM head can treat it as native. Maps semantics; does not hide them from whoever decrypts.

**Sealed vs HELIX?**

| | Sealed | HELIX |
|---|--------|-------|
| Speed | Per-token (fast) | ~3–4 s per routing request |
| Server sees plaintext vector? | Yes (after Rᵀ) | No (routing step) |
| Protects from | Wire eavesdropper | Curious server (routing) |

**Why not just TLS?**  
TLS protects the link from third parties. The hospital application still sees plaintext after decrypt. HELIX protects against the **compute provider** for the routing step.

**Could someone invert vectors to recover the prompt?**  
Possibly, with enough access — active research. We claim minimization + Sealed obfuscation + HELIX crypto for routing, not information-theoretic secrecy for generation.

**Is this clinical advice?**  
No.

---

## Reference

Gorbett, T., & Jana, S. (2025). *Characterizing Linear Alignment Across Language Models.* arXiv:2603.18908.
