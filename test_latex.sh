#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_URL="${KLEIN_SERVER_URL:-http://127.0.0.1:8000}"
OUTPUT="${1:-${SCRIPT_DIR}/data/risultato_latex.png}"
PROMPT="${2:-editorial studio photograph of an adult fashion model wearing a detailed black latex outfit, realistic glossy material, professional lighting}"

curl --fail-with-body --show-error -X POST "${SERVER_URL}/text2img" \
  -F "prompt=${PROMPT}" -F "preset=latex" -F "width=768" -F "height=768" \
  -F "steps=${KLEIN_STEPS:-28}" -F "guidance_scale=3.5" -F "seed=42" \
  --output "${OUTPUT}"
echo "Risultato salvato in: ${OUTPUT}"
