#!/usr/bin/env bash
# Start the full split demo: both backends + clinic + hospital frontends.
#
# Research prototype — not clinical product. Privacy story (honest):
#   1. Prompt stays on clinic (data minimization)
#   2. Sealed: rotated vectors on wire; hospital decrypts each step for generation
#   3. HELIX: CKKS routing only — hospital never sees plaintext vector or label
# See BUILD.md and hackathon_docs/demo-roadmap.md for setup and presentation.
#
# Usage (from repo root):
#   ./demo_split_all.sh
#
# Ctrl+C stops all four processes.
#
# --- Debugging: run each piece alone ---
# When something misbehaves (wrong-key demo, HELIX bootstrap, port conflict),
# separate terminals are easier — restart only what you need and read logs
# without four streams mixed together.
#
#   uv run python -m src.server --port 8420
#   uv run python -m src.practitioner_api --port 8421
#   cd frontend && npm run start:clinic     # http://localhost:4200
#   cd frontend && npm run start:hospital   # http://localhost:4201
#
# Backends only (no frontends):
#   bash scripts/demo_split.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SPECIALIST_PORT="${SPECIALIST_PORT:-8420}"
PRACTITIONER_PORT="${PRACTITIONER_PORT:-8421}"
CLINIC_PORT="${CLINIC_PORT:-4200}"
HOSPITAL_PORT="${HOSPITAL_PORT:-4201}"

PIDS=()

prefix_lines() {
  local tag="$1"
  while IFS= read -r line || [[ -n "$line" ]]; do
    printf '[%s] %s\n' "$tag" "$line"
  done
}

start_bg() {
  local tag="$1"
  shift
  "$@" 2>&1 | prefix_lines "$tag" &
  PIDS+=("$!")
}

cleanup() {
  echo ""
  echo "Shutting down split demo..."
  local pid
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  # Piped log wrappers may leave children; match known commands ([x] avoids pkill self-match)
  pkill -f "[u]v run python -m src.server --port ${SPECIALIST_PORT}" 2>/dev/null || true
  pkill -f "[u]v run python -m src.practitioner_api --port ${PRACTITIONER_PORT}" 2>/dev/null || true
  pkill -f "[n]g serve clinic" 2>/dev/null || true
  pkill -f "[n]g serve hospital" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=============================================="
echo "  CrossMind Split Demo — Full Stack"
echo "=============================================="
echo ""
echo "  Specialist (hospital API):  http://localhost:${SPECIALIST_PORT}"
echo "  Practitioner (clinic API):  http://localhost:${PRACTITIONER_PORT}"
echo "  Clinic UI:                  http://localhost:${CLINIC_PORT}"
echo "  Hospital UI:                http://localhost:${HOSPITAL_PORT}"
echo ""
echo "  Passphrase (both UIs): hackathon2026"
echo "  Sealed = rotation on wire (server decrypts for gen)"
echo "  HELIX = set key on clinic only; routing crypto (~3-4s), not full chat"
echo ""
echo "  Practitioner model load may take ~1 min — watch [practitioner] logs."
echo "  For debugging, run each service in its own terminal (see script header)."
echo "=============================================="
echo ""

start_bg "server" env PYTHONUNBUFFERED=1 uv run python -m src.server --port "$SPECIALIST_PORT"
start_bg "practitioner" env PYTHONUNBUFFERED=1 uv run python -m src.practitioner_api --port "$PRACTITIONER_PORT"
start_bg "clinic" npm --prefix "$ROOT/frontend" run start:clinic -- --port "$CLINIC_PORT"
start_bg "hospital" npm --prefix "$ROOT/frontend" run start:hospital -- --port "$HOSPITAL_PORT"

wait
