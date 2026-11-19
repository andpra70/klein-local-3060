#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
INPUT_IMAGE="${1:-${DATA_DIR}/test.jpg}"
OUTPUT_IMAGE="${2:-${DATA_DIR}/risultato.png}"
DEFAULT_PROMPT="mantieni aspetto del soggetto e dell'ambiente nel modo piu assoluto super fedele all'originale trasforma in stile line art shading tecniche antiche di stampa a mano con inchiostro e penna e matita e carboncino e pastello e acquerello e china e penna stilografica e pennello e pennarello e biro e gessetto e sanguigna e tempera e acrilico e olio su tela o carta o legno o metallo o vetro o plastica o ceramica o porcellana o marmo o pietra o cemento o mattoni o sabbia o terra o fango o neve o ghiaccio o acqua o fuoco o fumo "
PROMPT="${3:-${DEFAULT_PROMPT}}"
STEPS="${4:-${KLEIN_STEPS:-8}}"
RESIZED_IMAGE="$(mktemp --suffix=.png)"

cleanup() {
  rm -f -- "${RESIZED_IMAGE}"
}
trap cleanup EXIT

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Errore: immagine non trovata: ${INPUT_IMAGE}" >&2
  exit 1
fi

# Crea una copia temporanea massimo 512x512 mantenendo l'aspect ratio.
python -c '
import sys
from PIL import Image, ImageOps

source, destination = sys.argv[1:3]
with Image.open(source) as image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
    image.save(destination, format="PNG")
    print(f"Immagine test ridimensionata: {image.width}x{image.height}")
' "${INPUT_IMAGE}" "${RESIZED_IMAGE}"

curl --fail-with-body -X POST "http://127.0.0.1:8000/img2img" \
  -F "image=@${RESIZED_IMAGE};type=image/png" \
  -F "prompt=${PROMPT}" \
  -F "steps=${STEPS}" \
  -F "guidance_scale=1.0" \
  -F "seed=42" \
  --output "${OUTPUT_IMAGE}"

echo "Risultato salvato in: ${OUTPUT_IMAGE}"
