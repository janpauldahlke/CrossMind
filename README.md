# CrossMind

**One-line pitch:** We split sensitive text across two models: a local encoder and a remote specialist that works on vectors, not your sentence. Sealed obfuscates vectors on the wire; HELIX proves encrypted specialty routing where the math fits. Full encrypted generation is future work.

Research demo — not a clinical product. Based on [Gorbett & Jana, 2025](https://arxiv.org/pdf/2603.18908).



## Presentation Deck Online View: 

## ++[https://canva.link/mk9r9voee3948cl](https://canva.link/mk9r9voee3948cl)++

## Documentation


| Doc                                                                  | Use when                                                                  |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **[BUILD.md](BUILD.md)**                                             | Install, download models, train alignment, run server + UI                |
| **[hackathon_docs/overview.md](hackathon_docs/overview.md)**         | Understand architecture, privacy ladder, limits                           |
| **[hackathon_docs/pitch.md](hackathon_docs/pitch.md)**               | **June 6 judges** — 3 min pitch, 7 slides, submission                     |
| **[hackathon_docs/demo-roadmap.md](hackathon_docs/demo-roadmap.md)** | Longer live demo (lab / backup)                                           |
| **[hackathon_docs/spickzettel.md](hackathon_docs/spickzettel.md)**   | **Cheat sheet** — what runs where, alignment, LM head, Sealed, HELIX/CKKS |
| **[hackathon_docs/use_cases.md](hackathon_docs/use_cases.md)**       | Other industries (pilot sketches)                                         |


## Quick run (after [BUILD.md](BUILD.md))

```bash
./demo_split_all.sh
# Clinic  → http://localhost:4200
# Hospital → http://localhost:4201
# Passphrase both sides: hackathon2026
```

## Privacy in one table


| Step                   | Real gain                                         |
| ---------------------- | ------------------------------------------------- |
| Prompt stays on clinic | Data minimization — not encryption alone          |
| Alignment              | Maps spaces; hospital still gets vectors          |
| **Sealed** (rotation)  | Wire obfuscation — server decrypts for generation |
| **HELIX** (CKKS)       | Crypto for 5-class routing only (~3–4 s CPU)      |


Details: [overview.md](hackathon_docs/overview.md).

## Reference

Gorbett, T., & Jana, S. (2025). *Characterizing Linear Alignment Across Language Models.* arXiv:2603.18908.