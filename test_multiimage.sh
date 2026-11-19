#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SERVER_URL="${KLEIN_SERVER_URL:-http://127.0.0.1:8000}"
PERSON_IMAGE="${1:-${DATA_DIR}/mm.jpg}"
PANTS_IMAGE="${2:-${DATA_DIR}/pantaloni.jpg}"
OUTPUT_IMAGE="${3:-${DATA_DIR}/risultato_multiimage.png}"
STEPS="${4:-${KLEIN_STEPS:-8}}"
TEMP_DIR="$(mktemp -d)"
RESPONSE_FILE="${TEMP_DIR}/response.bin"

cleanup() {
  rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

for image in "${PERSON_IMAGE}" "${PANTS_IMAGE}"; do
  if [[ ! -f "${image}" ]]; then
    echo "Errore: immagine non trovata: ${image}" >&2
    exit 1
  fi
done

normalize_image() {
  local source="$1"
  local destination="$2"
  python -c '
import sys
from PIL import Image, ImageOps

with Image.open(sys.argv[1]) as source:
    image = ImageOps.exif_transpose(source).convert("RGB")
    image = ImageOps.pad(
        image, (512, 512), Image.Resampling.LANCZOS, color="white"
    )
    image.save(sys.argv[2], format="PNG")
' "${source}" "${destination}"
}

normalize_image "${PERSON_IMAGE}" "${TEMP_DIR}/person.png"
normalize_image "${PANTS_IMAGE}" "${TEMP_DIR}/pants.png"

PROMPT="Dress the person from image 1 with the short pants shown in image 2. Preserve the person's identity, face, body, pose and background. Make the shorts fit naturally and photorealistically. "

echo "Test multi-image: persona=${PERSON_IMAGE}, pantaloni=${PANTS_IMAGE}"
if ! curl --fail-with-body --silent --show-error \
    -X POST "${SERVER_URL}/multi-img2img" \
    -F "images=@${TEMP_DIR}/person.png;type=image/png" \
    -F "images=@${TEMP_DIR}/pants.png;type=image/png" \
    -F "prompt=${PROMPT}" \
    -F "width=512" \
    -F "height=512" \
    -F "steps=${STEPS}" \
    -F "guidance_scale=1.0" \
    -F "seed=42" \
    --output "${RESPONSE_FILE}"; then
  echo "Errore restituito dal server:" >&2
  python -m json.tool "${RESPONSE_FILE}" 2>/dev/null || cat -- "${RESPONSE_FILE}"
  exit 1
fi

python -c '
import sys
from PIL import Image
with Image.open(sys.argv[1]) as image:
    image.verify()
' "${RESPONSE_FILE}"

mv -- "${RESPONSE_FILE}" "${OUTPUT_IMAGE}"
echo "Risultato multi-image salvato in: ${OUTPUT_IMAGE}"
