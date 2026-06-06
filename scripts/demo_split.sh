#!/usr/bin/env bash
# Start the split demo stack (specialist + practitioner backends).
# Research prototype — clinic keeps prompt string; hospital gets vectors.
# Sealed: rotated wire obfuscation (server still decrypts). HELIX: routing HE only.
# Run clinic and hospital frontends in separate terminals:
#   cd frontend && npm run start:clinic
#   cd frontend && npm run start:hospital

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SPECIALIST_PORT="${SPECIALIST_PORT:-8420}"
PRACTITIONER_PORT="${PRACTITIONER_PORT:-8421}"

echo "=============================================="
echo "  CrossMind Split Demo — Backend Stack"
echo "=============================================="
echo ""
echo "  Specialist server:  http://localhost:${SPECIALIST_PORT}"
echo "  Practitioner API:   http://localhost:${PRACTITIONER_PORT}"
echo ""
echo "  Then in two more terminals:"
echo "    cd frontend && npm run start:clinic    → http://localhost:4200"
echo "    cd frontend && npm run start:hospital → http://localhost:4201"
echo ""
echo "  Demo passphrase (both sides): hackathon2026"
echo "  Wrong-key demo: use different passphrase on Hospital only"
echo "=============================================="
echo ""

cleanup() {
  echo ""
  echo "Shutting down backends..."
  kill "$SPECIALIST_PID" "$PRACTITIONER_PID" 2>/dev/null || true
  wait "$SPECIALIST_PID" "$PRACTITIONER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PYTHONUNBUFFERED=1 uv run python -m src.server --port "$SPECIALIST_PORT" &
SPECIALIST_PID=$!

PYTHONUNBUFFERED=1 uv run python -m src.practitioner_api --port "$PRACTITIONER_PORT" &
PRACTITIONER_PID=$!

wait
