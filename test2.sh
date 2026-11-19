#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_IMAGE="${1:-${DATA_DIR}/risultato_text2img.png}"
PROMPT="${2:-un gatto che tiene un cartello con la scritta hello world, fotografia cinematografica}"
TEMP_OUTPUT="$(mktemp --suffix=.png)"

cleanup() {
  rm -f -- "${TEMP_OUTPUT}"
}
trap cleanup EXIT

# Evita di scambiare un vecchio risultato per l'esito della richiesta corrente.
rm -f -- "${OUTPUT_IMAGE}"

curl --fail-with-body -X POST "http://127.0.0.1:8000/text2img" \
  -F "prompt=${PROMPT}" \
  -F "width=512" \
  -F "height=512" \
  -F "steps=4" \
  -F "guidance_scale=1.0" \
  -F "seed=42" \
  --output "${TEMP_OUTPUT}"

python -c '
import sys
from PIL import Image

with Image.open(sys.argv[1]) as image:
    image.verify()
print("PNG ricevuto e verificato")
' "${TEMP_OUTPUT}"

mv -- "${TEMP_OUTPUT}" "${OUTPUT_IMAGE}"

echo "Risultato text2img salvato in: ${OUTPUT_IMAGE}"
