#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVER_URL="${KLEIN_SERVER_URL:-http://127.0.0.1:8000}"
OUTPUT="${1:-${SCRIPT_DIR}/data/risultato_adult_latex.png}"
PROMPT="${2:-fine art studio photograph of a consenting adult age 30 wearing black latex, detailed natural anatomy, professional lighting, realistic skin and material texture}"

curl --fail-with-body --show-error -X POST "${SERVER_URL}/text2img" \
  -F "prompt=${PROMPT}" -F "preset=adult-latex" -F "width=768" -F "height=768" \
  -F "steps=${KLEIN_STEPS:-28}" -F "guidance_scale=3.5" -F "seed=42" \
  --output "${OUTPUT}"
echo "Risultato salvato in: ${OUTPUT}"
