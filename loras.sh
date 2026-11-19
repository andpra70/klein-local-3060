#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
SERVER_URL="${KLEIN_SERVER_URL:-http://127.0.0.1:8000}"
STEPS="${KLEIN_STEPS:-8}"
INPUT_IMAGE="${1:-${DATA_DIR}/test.jpg}"
OUTPUT_DIR="${2:-${DATA_DIR}/risultati_lora}"
TEMP_DIR="$(mktemp -d)"
CATALOG_FILE="${TEMP_DIR}/catalog.json"

cleanup() {
  rm -rf -- "${TEMP_DIR}"
}
trap cleanup EXIT

if [[ ! -f "${INPUT_IMAGE}" ]]; then
  echo "Errore: immagine di input non trovata: ${INPUT_IMAGE}" >&2
  exit 1
fi

mkdir -p -- "${OUTPUT_DIR}"

echo "Aggiornamento del catalogo LoRA da ${SERVER_URL}..."
if ! curl --fail-with-body --silent --show-error \
    "${SERVER_URL}/loras?refresh=true" \
    --output "${CATALOG_FILE}"; then
  echo "Errore restituito dal server durante l'aggiornamento del catalogo:" >&2
  if [[ -s "${CATALOG_FILE}" ]]; then
    python -m json.tool "${CATALOG_FILE}" 2>/dev/null || cat -- "${CATALOG_FILE}"
  fi
  exit 1
fi

# Risolve gli indici dalla fotografia corrente del catalogo. In questo modo lo
# script non dipende dall'ordine mutevole dei risultati di Hugging Face.
resolve_index() {
  local repo_id="$1"
  python -c '
import json
import sys

catalog_path, wanted_repo = sys.argv[1:3]
with open(catalog_path, encoding="utf-8") as stream:
    catalog = json.load(stream)
for item in catalog.get("items", []):
    if item.get("repo_id") == wanted_repo:
        print(item["index"])
        break
else:
    raise SystemExit(f"LoRA non presente nel catalogo: {wanted_repo}")
' "${CATALOG_FILE}" "${repo_id}"
}

PIXEL_REPO="Limbicnation/pixel-art-lora"
OUTPAINT_REPO="xocialize/outpaint-FLUX.2-klein-4B-lora"
OBJECT_REMOVE_REPO="xocialize/object-remove-FLUX.2-klein-4B-lora"
BACKGROUND_REMOVE_REPO="xocialize/background-remove-FLUX.2-klein-4B-lora"
SIM_TO_REAL_REPO="kphil01/flux.2-klein-4b-sim-to-real"
SPRITESHEET_REPO="xocialize/spritesheet-FLUX.2-klein-4B-lora"

PIXEL_INDEX="$(resolve_index "${PIXEL_REPO}")"
OUTPAINT_INDEX="$(resolve_index "${OUTPAINT_REPO}")"
OBJECT_REMOVE_INDEX="$(resolve_index "${OBJECT_REMOVE_REPO}")"
BACKGROUND_REMOVE_INDEX="$(resolve_index "${BACKGROUND_REMOVE_REPO}")"
SIM_TO_REAL_INDEX="$(resolve_index "${SIM_TO_REAL_REPO}")"
SPRITESHEET_INDEX="$(resolve_index "${SPRITESHEET_REPO}")"

REPOS=(
  "${PIXEL_REPO}"
  "${OUTPAINT_REPO}"
  "${OBJECT_REMOVE_REPO}"
  "${BACKGROUND_REMOVE_REPO}"
  "${SIM_TO_REAL_REPO}"
  "${SPRITESHEET_REPO}"
)
INDICES=(
  "${PIXEL_INDEX}"
  "${OUTPAINT_INDEX}"
  "${OBJECT_REMOVE_INDEX}"
  "${BACKGROUND_REMOVE_INDEX}"
  "${SIM_TO_REAL_INDEX}"
  "${SPRITESHEET_INDEX}"
)
FAILED_DOWNLOADS=()
FAILED_TESTS=()
SUCCESS_COUNT=0

echo "Download preventivo di tutti i LoRA..."
for position in "${!REPOS[@]}"; do
  echo "  [${INDICES[$position]}] ${REPOS[$position]}"
  download_response="${TEMP_DIR}/download-${position}.json"
  if ! curl --fail-with-body --silent --show-error \
      -X POST "${SERVER_URL}/loras/${INDICES[$position]}/download" \
      --output "${download_response}"; then
    echo "Errore restituito dal server durante il download di ${REPOS[$position]}:" >&2
    if [[ -s "${download_response}" ]]; then
      python -m json.tool "${download_response}" 2>/dev/null || cat -- "${download_response}"
    fi
    FAILED_DOWNLOADS+=("${REPOS[$position]}")
    echo "  Download fallito; il test verrà comunque tentato." >&2
  fi
done
echo "Fase di download completata. Avvio di tutti i test..."

# Normalizza l'immagine e crea gli input speciali richiesti dai LoRA di
# outpainting e object removal.
python -c '
import sys
from PIL import Image, ImageDraw, ImageOps

source, normalized_path, outpaint_path, highlighted_path = sys.argv[1:5]
with Image.open(source) as source_image:
    image = ImageOps.exif_transpose(source_image).convert("RGB")
    image = ImageOps.pad(
        image, (512, 512), Image.Resampling.LANCZOS, color="white"
    )
    image.save(normalized_path, format="PNG")

    inner = ImageOps.contain(image, (384, 384), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (512, 512), "#00ff00")
    canvas.paste(inner, ((512 - inner.width) // 2, (512 - inner.height) // 2))
    canvas.save(outpaint_path, format="PNG")

    highlighted = image.copy()
    draw = ImageDraw.Draw(highlighted, "RGBA")
    x0, y0 = image.width * 3 // 8, image.height * 3 // 8
    x1, y1 = image.width * 5 // 8, image.height * 5 // 8
    line_width = max(4, min(image.size) // 80)
    draw.rectangle((x0, y0, x1, y1), fill=(255, 0, 0, 90), outline=(255, 0, 0, 255), width=line_width)
    highlighted.save(highlighted_path, format="PNG")
' \
  "${INPUT_IMAGE}" \
  "${TEMP_DIR}/normal.png" \
  "${TEMP_DIR}/outpaint.png" \
  "${TEMP_DIR}/highlighted.png"

run_test() {
  local name="$1"
  local input="$2"
  local prompt="$3"
  local index="$4"
  local weight="$5"
  local weight_name="$6"
  local output="${OUTPUT_DIR}/${name}.png"
  local temporary_output="${TEMP_DIR}/${name}.png"
  local payload

  payload="$(python -c '
import json
import sys
index, weight, weight_name = sys.argv[1:4]
print(json.dumps({
    "index": int(index),
    "weight": float(weight),
    "parameters": {"weight_name": weight_name},
}))
' "${index}" "${weight}" "${weight_name}")"

  echo "Test ${name} con LoRA indice ${index}..."
  if ! curl --fail-with-body --silent --show-error \
      -X POST "${SERVER_URL}/img2img" \
      -F "image=@${input};type=image/png" \
      -F "prompt=${prompt}" \
      -F "steps=${STEPS}" \
      -F "guidance_scale=1.0" \
      -F "seed=42" \
      -F "lora=${payload}" \
      --output "${temporary_output}"; then
    echo "Errore restituito dal server durante il test ${name}:" >&2
    if [[ -s "${temporary_output}" ]]; then
      python -m json.tool "${temporary_output}" 2>/dev/null || cat -- "${temporary_output}"
    fi
    FAILED_TESTS+=("${name} | ${REPOS_BY_TEST[$name]}")
    rm -f -- "${temporary_output}"
    echo "  FALLITO: ${name}; continuo con il test successivo." >&2
    return 0
  fi

  if ! python -c '
import sys
from PIL import Image
with Image.open(sys.argv[1]) as image:
    image.verify()
' "${temporary_output}"; then
    FAILED_TESTS+=("${name} | ${REPOS_BY_TEST[$name]} | risposta non valida")
    rm -f -- "${temporary_output}"
    echo "  FALLITO: ${name} ha restituito un'immagine non valida; continuo." >&2
    return 0
  fi
  mv -- "${temporary_output}" "${output}"
  SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  echo "  Risultato: ${output}"
}

declare -A REPOS_BY_TEST=(
  ["01-pixel-art"]="${PIXEL_REPO}"
  ["02-outpaint"]="${OUTPAINT_REPO}"
  ["03-object-remove"]="${OBJECT_REMOVE_REPO}"
  ["04-background-remove"]="${BACKGROUND_REMOVE_REPO}"
  ["05-sim-to-real"]="${SIM_TO_REAL_REPO}"
  ["06-spritesheet"]="${SPRITESHEET_REPO}"
)

run_test \
  "01-pixel-art" "${TEMP_DIR}/normal.png" \
  "transform the subject into a detailed 16-bit pixel art game character, preserve pose and identity" \
  "${PIXEL_INDEX}" "0.9" "pytorch_lora_weights.safetensors"

run_test \
  "02-outpaint" "${TEMP_DIR}/outpaint.png" \
  "Fill the green spaces according to the image" \
  "${OUTPAINT_INDEX}" "1.1" "flux-outpaint-lora.safetensors"

run_test \
  "03-object-remove" "${TEMP_DIR}/highlighted.png" \
  "Remove the highlighted object from the scene" \
  "${OBJECT_REMOVE_INDEX}" "1.1" "flux-object-remove-lora.safetensors"

run_test \
  "04-background-remove" "${TEMP_DIR}/normal.png" \
  "Remove the background from the image" \
  "${BACKGROUND_REMOVE_INDEX}" "1.1" "flux-background-remove-lora.safetensors"

run_test \
  "05-sim-to-real" "${TEMP_DIR}/normal.png" \
  "Turn this scene into a real photo" \
  "${SIM_TO_REAL_INDEX}" "1.0" "flux2-klein-4b-sim-to-real.safetensors"

run_test \
  "06-spritesheet" "${TEMP_DIR}/normal.png" \
  "2x2 sprite sheet" \
  "${SPRITESHEET_INDEX}" "1.1" "flux-spritesheet-lora.safetensors"

echo
echo "Riepilogo LoRA: ${SUCCESS_COUNT}/6 test riusciti."
echo "Risultati validi disponibili in: ${OUTPUT_DIR}"

if (( ${#FAILED_DOWNLOADS[@]} > 0 )); then
  echo "Download falliti (${#FAILED_DOWNLOADS[@]}):" >&2
  printf '  - %s\n' "${FAILED_DOWNLOADS[@]}" >&2
fi

if (( ${#FAILED_TESTS[@]} > 0 )); then
  echo "Test falliti (${#FAILED_TESTS[@]}):" >&2
  printf '  - %s\n' "${FAILED_TESTS[@]}" >&2
  exit 1
fi

echo "Tutti i test LoRA sono terminati correttamente."
