# Frontend

Angular 19 workspace with two applications for the split demo.

## Applications

| App | Port | Backend | Purpose |
| --- | ---- | ------- | ------- |
| **clinic** | 4200 | practitioner_api :8421 | Patient-facing clinic UI (Sealed + HELIX) |
| **hospital** | 4201 | server :8420 | Hospital specialist UI (wire view, compute) |

## Development

```bash
npm install

# Start both (or use demo_split_all.sh from repo root):
npm run start:clinic
npm run start:hospital
```

## Build

```bash
npm run build:clinic
npm run build:hospital
```
