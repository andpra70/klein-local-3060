#!/usr/bin/env bash

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "flux-klein" ]]; then
  echo "Errore: attiva prima l'ambiente con: conda activate flux-klein" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec python server_klein.py

