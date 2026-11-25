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
  ./go.sh [BACKEND] [MODALITA]

Backend disponibili:
  klein    Predefinito. FLUX.2 Klein 4B, veloce e con multi-image.
  flux1    FLUX.1-dev, compatibile con i preset adult/anatomy/latex.

Modalità disponibili:
  int8     Predefinita. Transformer e text encoder in INT8.
           Più veloce per img2img (~2 s sulla RTX 3060), ma text2img può
           produrre immagini corrotte o illeggibili.

  nf4      Transformer NF4 a 4 bit e text encoder FP16, tutto in VRAM.
           Con FLUX.1 anche il secondo text encoder viene quantizzato per
           rientrare nella VRAM. Compromesso consigliato sulla RTX 3060.

  offload  Modello BF16 senza quantizzazione, con CPU offload.
           Qualità di riferimento per img2img e text2img, ma molto lento
           (~44 s) a causa dei trasferimenti tra RAM e GPU.

Esempi:
  ./go.sh                    # equivale a: ./go.sh klein int8
  ./go.sh klein nf4
  ./go.sh flux1 nf4          # consigliato per RTX 3060 12 GB
  ./go.sh flux1 offload      # più lento, usa più RAM di sistema
  ./go.sh --help

I tempi sono indicativi e derivano dai test su RTX 3060 12 GB.
EOF
}

LOAD_MODE="int8"
MODE_SET=0
BACKEND="klein"
BACKEND_SET=0

for argument in "$@"; do
  case "${argument}" in
    -h|--help|help)
      show_help
      exit 0
      ;;
    int8|nf4|offload)
      if (( MODE_SET )); then
        echo "Errore: specificare una sola modalità." >&2
        show_help >&2
        exit 2
      fi
      LOAD_MODE="${argument}"
      MODE_SET=1
      ;;
    klein|flux1)
      if (( BACKEND_SET )); then
        echo "Errore: specificare un solo backend." >&2
        exit 2
      fi
      BACKEND="${argument}"
      BACKEND_SET=1
      ;;
    *)
      echo "Errore: argomento non valido: ${argument}" >&2
      show_help >&2
      exit 2
      ;;
  esac
done

activate_conda_environment
cd "${SCRIPT_DIR}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export KLEIN_LOAD_MODE="${LOAD_MODE}"
export KLEIN_BACKEND="${BACKEND}"

echo "Ambiente Conda: ${CONDA_DEFAULT_ENV}"
echo "Backend: ${KLEIN_BACKEND}"
echo "Modalità: ${KLEIN_LOAD_MODE}"

exec python server_klein.py

