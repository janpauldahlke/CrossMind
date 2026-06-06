# Demo artifacts (`pairs`)

| File | In git? | Purpose |
| ---- | ------- | ------- |
| `alignment_map.npz` | **Yes** | Linear map `W*`, `b*` (Qwen hidden dim -> Llama hidden dim) |
| `routing_head.npz` | **Yes** | HELIX 5-class routing weights |
| `lm_head.npy` | **No** (~2 GB) | Llama vocabulary projection — run `train_alignment.py` once to extract, or copy from an existing training run |
| `tokenizer/` | **Yes** | Llama tokenizer files |

**After clone:** if the split demo fails on "missing lm_head", run:

```bash
uv run python scripts/train_alignment.py \
  --practitioner-id qwen-2.5-7b \
  --specialist-id llama-3.1-8b
```

That writes `lm_head.npy` locally (not committed — too large for GitHub).
