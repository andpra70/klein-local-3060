#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="flux-klein"

activate_conda_environment() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "${CONDA_ENV}" ]]; then
    return
  fi

  local conda_base=""
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
  else
    local candidate
    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda"; do
      if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
        conda_base="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${conda_base}" || ! -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
    echo "Errore: Conda non trovato. Installa Miniconda o rendi disponibile il comando conda." >&2
    exit 1
  fi

  # shellcheck disable=SC1091
  source "${conda_base}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
}

show_help() {
  cat <<'EOF'
Uso:
  ./go.sh [MODALITA]

Modalità disponibili:
  int8     Predefinita. Transformer e text encoder in INT8.
           Più veloce per img2img (~2 s sulla RTX 3060), ma text2img può
           produrre immagini corrotte o illeggibili.

  nf4      Transformer NF4 a 4 bit e text encoder FP16, tutto in VRAM.
           Compromesso consigliato per usare sia img2img (~6 s) sia text2img
           con qualità valida. NF4 richiede dequantizzazione durante il calcolo.

  offload  Modello BF16 senza quantizzazione, con CPU offload.
           Qualità di riferimento per img2img e text2img, ma molto lento
           (~44 s) a causa dei trasferimenti tra RAM e GPU.

Esempi:
  ./go.sh             # equivale a: ./go.sh int8
  ./go.sh nf4
  ./go.sh offload
  ./go.sh --help

I tempi sono indicativi e derivano dai test su RTX 3060 12 GB.
EOF
}

if (( $# > 1 )); then
  echo "Errore: specificare al massimo una modalità." >&2
  show_help >&2
  exit 2
fi

case "${1:-int8}" in
  -h|--help|help)
    show_help
    exit 0
    ;;
  int8|nf4|offload)
    LOAD_MODE="${1:-int8}"
    ;;
  *)
    echo "Errore: modalità non valida: $1" >&2
    show_help >&2
    exit 2
    ;;
esac

activate_conda_environment
cd "${SCRIPT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export KLEIN_LOAD_MODE="${LOAD_MODE}"

echo "Ambiente Conda: ${CONDA_DEFAULT_ENV}"
echo "Avvio Klein in modalità: ${KLEIN_LOAD_MODE}"

exec python server_klein.py

