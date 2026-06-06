# CrossMind — industry use cases

**CrossMind** is a way for two organizations to use language models together on sensitive text. The **local site** runs a smaller model and keeps the original words on its own machines. The **central hub** runs the stronger model’s output layer (generation) or a small classifier (routing). Between them travels a **hidden vector** — not the full document string.

The approach builds on published research showing that different large language models can share a simple **linear map** between their internal representations ([Gorbett & Jana, 2025](https://arxiv.org/pdf/2603.18908)). The repo implements that map, a live two-party demo, **Sealed** rotation on the wire, and **HELIX** homomorphic routing (5 classes).

**Context:** [overview.md](overview.md) · **Setup:** [BUILD.md](../BUILD.md)

This document is for **judges and design partners**. Each industry uses the same shape: **setting, problem, risks, proposal, pilot**. Sections marked **Pilot** are **aspirational design-partner sketches** — not shipped product. Our hackathon demo is **healthcare** (implemented in repo).

---

## Privacy ladder (same in every vertical)

| Step | What happens | Who learns what | Real privacy gain? |
| ---- | ------------ | --------------- | ------------------ |
| **1. Prompt stays local** | Edge model encodes text; hub gets no prompt string | Hub disk/logs don’t store raw narrative | Data minimization |
| **2. Alignment** | Linear map to specialist space | Hub still receives vectors | Maps semantics; doesn’t hide from decryptor |
| **3. Sealed** | Rotation on wire; hub reverses R | Eavesdropper blocked; **hub decrypts for generation** | Wire obfuscation |
| **4. HELIX** | CKKS matmul on routing head | Hub no plaintext vector/label for that step | Crypto for routing only |
| **5. Hub UI / logs** | Metadata, timing | Demo visibility | Not a security boundary |

**Honest scope:** Research demo + architecture path. Not certified compliance. Routing ~83% val accuracy in our medical head. Full encrypted generation not practical today.

---

## What CrossMind does in one picture

Many teams face the same tension: they want the **best central model**, but policy says they **cannot ship raw text** to that model’s cloud.


| What organizations say | What CrossMind does (honest) |
| ---------------------- | ---------------------------- |
| “We cannot send the full story to headquarters.” | Local site turns text into a vector locally; hub gets **numbers**, not the sentence string. |
| “We still need headquarters’ model quality.” | Hub keeps specialist LM head (generation) or routing head (classification). |
| “Someone might tap the network link.” | **Sealed**: shared secret **rotates** the vector — protects **passive eavesdropper**, not curious hub. |
| “Headquarters must not see what we encoded (for routing).” | **HELIX**: hub multiplies on ciphertext for **5-class routing**; only local site decrypts label (**implemented in repo**). |


**Working in the repo today:** local Qwen encodes → alignment → Sealed rotation → remote Llama LM head → streamed text; split clinic/hospital UI; HELIX encrypted routing (~3–4 s CPU).

**Not in repo / not claimed:** production deployment, compliance certification, full HE chat, encrypted generation at vocab scale.

---

## Healthcare (live hackathon demo — implemented)

**Setting.** A **clinic** keeps the chart and runs a smaller on-premise language model. A **hospital group or specialist service** runs the larger model’s prediction layer. The specialist side **does not receive the prompt string** — it receives numeric vectors (which still encode information).

**Problem.** Clinicians want help that feels like asking a strong specialist model, but security teams push back when every symptom sentence is uploaded as plain text. Regulations (e.g. GDPR, US health privacy rules) often treat the cloud provider as a **processor** when they receive identifiable narrative — we **aim at minimization and selective HE**, not “HIPAA compliant out of the box.”

**Risks if nothing changes.**

- Full-text export increases audit scope, breach impact, and vendor dependency.
- One giant cloud model forces all sites to accept the same data flow.
- Products marketed as “encrypted chat” often still **decrypt at the provider** for generation — which security reviews correctly question.

**Our proposal (demo).**

- **Architectural split:** clinic encodes locally; hospital gets vectors per step, not prompt text.
- **Sealed generation:** rotated vectors on the wire; hospital **decrypts every step** to generate — obfuscation vs eavesdropper, not vs curious hospital.
- **HELIX routing:** homomorphic 5-department scoring (~83% val accuracy); hospital never sees plaintext vector or label for that request.

**Pilot (design partner sketch — ~90 days, not shipped).** One clinic–hospital pair, one model pair, measure routing vs plaintext baseline, document what each side logs. All clinical wording **human-reviewed**; supports triage/drafting research only, not autonomous diagnosis.

---

## Insurance — group claims intelligence

*Pilot sketch — aspirational design partner scenario.*

**Setting.** A **regional insurer** handles claims locally. A **group hub** has models trained on history from many subsidiaries. Subsidiaries cannot dump full claim narratives into the group lake.

**Problem.** Adjusters need group expertise (fraud desk, medical review, catastrophe, reinsurance) without exporting stories the subsidiary forbids.

**Our proposal (same CrossMind pattern).** Regional office encodes locally; group hub receives aligned vectors — **not original paragraphs**. **Sealed** protects the WAN link from passive tapping. **HELIX-style routing** could score into buckets without the hub decrypting the encoding for that step (would require custom routing head + integration).

**Pilot sketch.** Closed claims with known outcomes; compare routing to plaintext baseline; audit logs showing narratives never left regional env. Generated text through adjuster approval only.

---

## Legal — law firm and group litigation analytics

*Pilot sketch — aspirational design partner scenario.*

**Setting.** A **law firm** holds privileged matter text. A **group analytics** vendor has cross-matter models. The firm cannot upload matter text as ordinary clear text.

**Problem.** Partners want staffing/escalation signals and research help under **outside-counsel** and ethics constraints on cloud LLMs.

**Our proposal.** Firm encodes an **internal summary** locally; hub returns **routing/scoring** (or Sealed-assisted draft bullets for partner review only). **Sealed** on the link; **HELIX** where the hub must not retain readable encodings for classification.

**Pilot sketch.** Closed matters with partner-confirmed labels; ethics and infosec review of retention. Not autonomous legal advice.

---

## Industrial chemistry — plant and corporate process intelligence

*Pilot sketch — aspirational design partner scenario.*

**Setting.** A **plant** holds proprietary batch narratives. **Corporate** HSE/manufacturing excellence has cross-site models. Plants won’t upload full batch stories.

**Problem.** Shift teams want corporate playbooks when a batch “looks like” past events — without sending recipe and operator notes as files.

**Our proposal.** Plant encodes a **short batch summary** locally; corporate receives aligned vectors. Returns a **route** (normal, process dev, HSE, quality hold, environmental). Where policy requires, **HELIX** scoring on encrypted vectors. Any operator hints **advisory only** — DCS/SIS remain authoritative.

**Pilot sketch.** One site, one line, parallel with existing SPC; OT approves fixed outbound message types.

---

## For founders — how this becomes a business

CrossMind is **B2B infrastructure**, not a consumer app. First deals are **design partners** for a focused pilot: one model pair, one routing table, honest security documentation, workflow hooks.

**Near-term credible product story:** routing and scoring with **minimal text export** + **HELIX** where linear classification fits. Cross-model **Sealed** generation is a compelling **research demo** of alignment science; it is **not** “encrypted ChatGPT” today.

Expansion: more categories, more model pairs, workflow integration — **not** full-vocab HE generation on current hardware without major research progress.
