# CrossMind — pitch deck (7 slides)

**Use:** AI BEAVERS founder hackathon · June 6 · **3 minutes** live  
**Goal:** Showcase the idea honestly · meet people · not “win or die”  
**Export:** Copy each slide block into **[Slides.com](https://slides.com)** (web, works anywhere) or Google Slides / Keynote. One idea per slide. No paragraphs on screen.

**Submission:** export **PDF** from Slides.com (Share → Export → PDF) for the hackathon form by 19:00.

**Speaker notes** below each slide = **what you say on stage**.  
A **private section at the end** = your real goals + what to do if you win but can’t take a prize — **never say that on stage**.

---

## Design tips (you said you’re not a slides person)

- **Tool:** [Slides.com](https://slides.com) — log in once, present from any machine; **Speaker Notes** panel = paste the “Speaker notes” blocks below each slide
- **Template:** pick a minimal theme (dark + one accent, or clean light); avoid busy hackathon templates
- **Font:** one family only (Slides default or Inter). Title 32–36 pt, body 20–24 pt
- **Rule:** max **6 words** in the headline; max **4 bullets** per slide
- **Proof:** slide 2 = **screenshot** of clinic + hospital UI side by side (biggest win)
- **Footer on every slide:** `DEMO — research prototype · not medical advice`
- **QR (optional):** slide 2 → GitHub repo URL

---

## Slide 1 — Problem + customer

### On screen

**Headline:**  
Central LLM blocked. Local model too small.

**Bullets:**

- **Buyer:** hospital-group IT / clinic compliance (multi-site)
- **Pain:** want specialist AI without raw chart text on central GPU
- **Today:** ban cloud LLM *or* full-text API export

**Visual:** Simple split — left “📋 full note → ☁️” crossed out; right “?”

### Speaker notes (~25 s) — **on stage**

> “In regulated healthcare — our demo vertical — hospital-group IT often blocks uploading patient notes to a central LLM. Clinics still want specialist-grade AI. Today the choice is **no AI** or **export the full chart**. CrossMind is a working prototype for a third path: **split inference** with honest security boundaries.”

---

## Slide 2 — Solution + demo

### On screen

**Headline:**  
CrossMind — split inference, honest crypto

**Bullets:**

- **Clinic:** Qwen encodes locally — **prompt never sent as text**
- **Hospital:** Llama decodes from **vectors** (Sealed rotation on wire)
- **HELIX:** homomorphic **5-department routing** (~4 s) — not full chat

**Visual:** **Screenshot** — clinic UI + hospital wire tab, tiled. Label arrows: “text stays here” / “packets only”

**Small line:** `github.com/…` or QR

### Speaker notes (~40 s)

> “**CrossMind**: small model on the clinic, specialist model on the hospital — they share **numbers**, not your sentence. For generation we use **Sealed**: rotated vectors on the wire; the hospital **still decrypts each step** to generate — that’s obfuscation against a wire tap, not magic privacy from the hospital.  
> For **routing** — cardiology, neurology, oncology, orthopedics, general — we use **HELIX**: real CKKS homomorphic encryption on a five-class head. The hospital sees a heatmap, **not** the department name. Clinic decrypts.  
> Live demo on my laptop if we have time — or grab the repo.”

*If demo fails:* “Repo and screen recording — the split UI is the proof.”

---

## Slide 3 — Why now (not a wrapper)

### On screen

**Headline:**  
Why this is possible now

**Bullets:**

- Linear **cross-model alignment** — published, reproducible (Gorbett & Jana 2025)
- **CKKS** fits **small linear heads** — routing in ~4 s, not 128k vocab
- Buyers already ask for **data minimization** — not “trust our SaaS”

**Visual:** Tiny diagram: `Qwen h → W* → Llama head` (copy from README, simplify)

**Do not write:** “HIPAA compliant” · “end-to-end encrypted LLM”

### Speaker notes (~25 s)

> “This isn’t ChatGPT with a hospital skin. Different models live in different vector spaces — recent work shows you can **align** them with a linear map. Homomorphic encryption still can’t do full chat at practical speed, but a **five-logit routing head** is credible today. And compliance teams already think in **minimization**, even when they’re not buying crypto.”

---

## Slide 4 — Landscape (no $50B slide)

### On screen

**Headline:**  
Alternatives today

**Bullets:**

| Option | Tradeoff |
|--------|----------|
| Block AI entirely | Safe, no capability |
| Full-text cloud LLM | Capability, audit / policy risk |
| On-prem only | No shared specialist |
| **CrossMind (demo)** | Vectors + selective HE for routing |

**Visual:** 2×2 or simple table (above fits as table on slide)

### Speaker notes (~20 s)

> “Competitors aren’t only other startups — it’s **do nothing** and **email the PDF to headquarters**. We’re not claiming to replace Epic or to be compliant out of the box. We’re showing **infrastructure**: encode locally, decode remotely, encrypt **where the math fits**.”

---

## Slide 5 — Evidence (honest)

### On screen

**Headline:**  
What exists today

**Bullets:**

- Working **split UI** + CLI two-party demo
- Alignment **~0.83** cosine (trained pair, public corpus)
- Routing head **~83%** val accuracy — demo prompts clear-cut
- **Research prototype** — not clinical validation

**Visual:** One metric callout box: `83% routing val · ~0.83 alignment cosine`

**Optional (if true):** one line: “Discussed with [role/org]” — **only if real**

### Speaker notes (~25 s) — **on stage**

> “The demo runs end-to-end: split UI, metrics in the repo — alignment cosine about **0.83**, routing about **eighty-three percent** on held-out labels. That’s enough to prove the **architecture**, not enough to claim clinical-grade triage. Footer on every slide: **research prototype**, not medical advice.”

---

## Slide 6 — Go-to-market wedge

### On screen

**Headline:**  
Pilot path

**Bullets:**

- Wedge: **routing + draft assist** with data minimization — not “private ChatGPT”
- First **2–3 design-partner** hospital groups — one model pair, security documentation
- Expand categories and model pairs after routing accuracy validated on partner data

**Visual:** None needed — keep text clean

### Speaker notes (~20 s) — **on stage**

> “We’d start narrow: one hospital-group pilot — encrypted **department routing** plus Sealed cross-model drafting, with a clear data-flow audit. No compliance certification claims — **minimization and selective homomorphic encryption** where linear math fits. Prove routing on partner labels, then widen.”

---

## Slide 7 — Team + next step

### On screen

**Headline:**  
Built solo · recruiting partners

**Bullets:**

- **[Your name]** — full-stack · paper → working split demo in 8 weeks
- **Honest limit:** Sealed gen needs hospital decrypt; HELIX covers **routing only**
- **Next:** design-partner conversations with hospital IT / clinical informatics

**Contact:** email · GitHub · LinkedIn

**Visual:** Photo optional — many skip for hackathon

### Speaker notes (~25 s) — **on stage**

> “I built this **solo** — from alignment paper to live two-party demo with Sealed generation and HELIX routing. The weak point I’d test next: routing accuracy on real partner data, and how far minimization gets you before buyers ask for full HE. I’m looking for **design-partner intros** in hospital IT or informatics. Happy to show the demo after. Thank you.”

---

## 3-minute timer (total ~180 s)

| Slide | Seconds |
|-------|---------|
| 1 Problem | 25 |
| 2 Solution + demo hint | 40 |
| 3 Why now | 25 |
| 4 Landscape | 20 |
| 5 Evidence | 25 |
| 6 Hypothesis | 20 |
| 7 Team + ask | 25 |

Practice with phone timer. Cut slide 4 or 6 first if you run long.

---

## What to say when networking (hallway — more candid OK)

**Stage version** (still confident):

> “Split LLM inference — local encode, remote specialist on vectors, homomorphic encryption for medical routing. Working demo on my laptop.”

**Hallway version** (if you trust the person — never on stage):

> “Research showcase — I’m here to meet people who’ve blocked central LLM rollouts. Not shopping for co-founders tonight, but always up for a sharp conversation.”

---

## Private — for you only (never on stage)

Your real goals (network, learn, showcase) are fine. **Play the game in the room:** pitch like a credible builder with a wedge and a pilot path. You can still be **honest on tech** (research prototype, Sealed vs HELIX, no HIPAA) — that reads as strength, not sabotage.

**Do not say on stage:**

- “I have a 9–5 and wouldn’t quit to found this”
- “This isn’t my startup / I’d rather build Rust agents”
- “I’m not trying to win” / “I can’t accept a prize”
- “I’m not asking for investment” (sounds like you don’t believe in it)

**If you win but cannot accept a prize** (employment policy, sponsor rules, etc.):

1. **On stage:** smile, thank organizers, take the moment — you don’t owe the room your employment details.
2. **After:** email **hi@ai-beavers.com** or find an organizer privately: grateful, proud of the demo, **must decline or transfer** the prize per your situation — ask what they allow (decline, donate to another team, forfeit credits).
3. Many events prefer a graceful decline to a public “I can’t take this” on the mic.

You’re not cheating by pitching well while knowing your constraints. You’d be cheating if you **lied** about the product (HIPAA, encrypted ChatGPT, fake traction). Technical honesty + confident delivery is the right combo.

---

## Prizes (what people actually chase)

AI BEAVERS **doesn’t publish fixed cash amounts** in the participant guide — main prizes are typically **finalist visibility**, **sponsor tracks** (OpenAI, Cursor, Qwen, etc. — credits/tools announced in Discord `#announcements`), and **investor/angel attention** for a subset of teams.

Many attendees optimize for **sponsor credits** ($500–5k API tokens is common at similar events — exact amounts vary by sponsor and year). That’s fine; your goal (**showcase + network**) is aligned with doing a **memorable demo** and a **honest 3 minutes**, not maximizing token coupons.

---

## Before you present — checklist

- [ ] Slide 2 screenshot from `./demo_split_all.sh` (Sealed + HELIX if possible)
- [ ] Replace `[Your name]` and contact on slide 7
- [ ] Optional: fill slide 5 “Discussed with …” **only if true**
- [ ] Read [demo-roadmap.md § Banned phrases](demo-roadmap.md#banned-phrases) once
- [ ] Export **PDF** from Slides.com for submission form by **19:00**
- [ ] Demo laptop: passphrase `hackathon2026`, HELIX key set on clinic only

---

## Copy-paste: slide titles only (for outline mode)

1. Central LLM blocked. Local model too small.
2. CrossMind — split inference, honest crypto
3. Why this is possible now
4. Alternatives today
5. What exists today
6. Pilot path
7. Built solo · recruiting partners
