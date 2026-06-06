# Pitch guide — AI BEAVERS founder hackathon (June 6)

**Event constraints:** 3 minutes live pitch (preliminary); finalists get 3 + 2 Q&A. Deck **max 7 slides**. Submit public GitHub repo + deck by **19:00**.

Logistics: [founder hackathon - participant guide (logistics).md](founder%20hackathon%20-%20participant%20guide%20%28logistics%29.md)  
Strategy (external): AI BEAVERS Ideas & Strategy guide.

This doc maps **CrossMind** to what judges actually score — not an 8-minute lab demo (see [demo-roadmap.md](demo-roadmap.md) for that).

---

## First 15 seconds (memorize)

> “**Hospital-group IT** blocks uploading patient notes to a central LLM. We let the **clinic keep the chart locally** and send only **aligned vectors** to a **specialist model** — plus **real homomorphic encryption** for **five-department routing**, not full chat. Research demo today; wedge is **split inference + selective HE where linear math fits**.”

Then: one sentence founder edge + one honest limit.

---

## Seven slides (deck outline)

| # | Slide | CrossMind content |
| --- | ----- | ----------------- |
| 1 | **Problem + customer** | Buyer: **hospital-group AI lead / clinic IT + compliance** at a multi-site group. Pain: want Llama-class help without raw EMR text on the specialist GPU or in vendor logs. Status quo: “no cloud LLM” policy, or full-text API (audit nightmare). |
| 2 | **Solution + product** | Split: Qwen on clinic, Llama head on hospital; **Sealed** wire obfuscation; **HELIX** CKKS routing (5 specialties). Screenshot: clinic + hospital UI side by side. Demo: `./demo_split_all.sh` or hosted URL if you have one. |
| 3 | **Why now** | (1) Linear cross-model alignment is published and reproducible ([Gorbett & Jana, 2025](https://arxiv.org/pdf/2603.18908)). (2) CKKS can run a **5-class head** in ~4 s — credible crypto hook, not 128k vocab. (3) Regulated buyers already ask for **minimization**, not “trust our SaaS.” **Not** “we wrapped ChatGPT.” |
| 4 | **Market + competition** | Bottom-up: N hospital groups × pilot fee — don’t lead with $50B healthcare AI. Competitors: full-text cloud LLM APIs, on-prem only (no central specialist), federated learning (slow). **Status quo:** block AI entirely. Edge: **cross-vendor model split + HE routing** in one demoable stack. |
| 5 | **Business model + evidence** | Wedge product: **routing + draft assist with minimization**, not encrypted ChatGPT. Evidence to bring: demo video, alignment metrics (~0.83 cosine), routing val ~83%, one design-partner conversation / domain insight — **honest, even if small**. No fake traction. |
| 6 | **Go-to-market** | Narrow pilot: routing + draft assist with minimization — 2–3 design-partner hospital groups |
| 7 | **Team + next step** | Solo builder, paper → demo; honest limits; seeking design-partner intros |

---

## 3-minute talk track

| Time | Say |
| ---- | --- |
| 0:00–0:20 | Hook (above) + buyer |
| 0:20–1:00 | Problem + status quo (spreadsheet/policy “no AI”) |
| 1:00–1:45 | Live or screenshot: Sealed gen — **no prompt on hospital UI**; vectors only |
| 1:45–2:15 | HELIX: hospital sees heatmap, **not** department; clinic decrypts label |
| 2:15–2:45 | Why now + wedge (routing/minimization, not wrapper) |
| 2:45–3:00 | Weakest point + “research demo, path to pilot” |

**Do not** use 8 minutes of [demo-roadmap.md](demo-roadmap.md) on stage — judges will cut you off.

---

## Reality check (strategy guide → our answers)

| Question | CrossMind answer (honest) |
| -------- | ------------------------- |
| **Wedge (one sentence)** | Split sensitive text across two models; HE only for **5-class routing**. |
| **Buyer** | Hospital-group **AI/platform lead** or **clinic IT** blocking raw chart export. |
| **Demand** | Research demo — bring **best real signal** (pilot interest, informatics contact, not “judges said cool”). |
| **Status quo** | Ban central LLM **or** ship full text to vendor API. |
| **Why now** | Alignment paper + practical CKKS for small linear heads + buyer pressure on minimization. |
| **Founder-market fit** | *Fill in your sentence* — e.g. lived hospital IT policy pain, ML + security background. |
| **Evidence** | Metrics in repo + demo; add **one non-copyable** thing before June 6 if you can. |

---

## Avoid weak-idea patterns (how we read)

| Pattern | CrossMind risk | Reframe |
| ------- | -------------- | ------- |
| “ChatGPT for healthcare” | High if you lead with chat | Lead with **buyer + minimization + routing HE** |
| Feature not company | HELIX alone looks small | **Infrastructure**: split inference protocol + selective HE hooks |
| Big TAM slide | Judges hate it | Bottom-up pilot economics |
| “HIPAA compliant” | Disqualifying overclaim | **Minimization + selective HE — not certified** |
| Platform / ecosystem | Vague | **One workflow:** encode locally → specialist decode → optional encrypted route |

---

## Submission checklist (19:00)

- [ ] Public repo — README points to [overview.md](overview.md) + [BUILD.md](../BUILD.md)
- [ ] **Meaningful June 6 commits** if rules require day-of build (see logistics guide — don’t misrepresent pre-event work)
- [ ] Deck ≤ 7 slides (outline above)
- [ ] Optional: hosted demo URL or “run locally” QR to BUILD.md
- [ ] Pitch practiced at **3 minutes** out loud
- [ ] Banned phrases reviewed — [demo-roadmap.md § Banned phrases](demo-roadmap.md#banned-phrases)

---

## If you already had this repo before June 6

Organizers allow **market knowledge**, not submitting an old product as new work. Fair approach:

- **June 6 scope:** one sharp slice (e.g. split UI flow, HELIX demo polish, wrong-key narrative) with **visible commits that day**
- **Pitch:** “We sharpened the wedge live” — don’t claim you trained alignment in 12 hours
- **Deck:** problem/customer from real knowledge; demo shows what you **built or integrated for this room**

See logistics guide § “Already have a startup?”

---

## Judge Q&A prep (finalists)

Short honest answers — expand from [overview.md § FAQ](overview.md#faq):

- **Why not one model on-prem?** Cost, updates, best specialist without colocating raw charts.
- **Does the hospital learn nothing?** No — vectors encode information; Sealed server decrypts for generation; HELIX hides routing step only.
- **Why not TLS?** TLS ≠ hiding data from the compute provider.
- **Clinical use?** Research prototype; not diagnosis; human review required.

**Slide copy + speaker notes:** [deck.md](deck.md)
