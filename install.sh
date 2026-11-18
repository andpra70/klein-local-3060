#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "flux-klein" ]]; then
  echo "Errore: attiva prima l'ambiente Conda:" >&2
  echo "  conda activate flux-klein" >&2
  exit 1
fi

echo "Installazione delle dipendenze nell'ambiente: ${CONDA_DEFAULT_ENV}"
python -m pip install --upgrade pip
python -m pip install --requirement "${SCRIPT_DIR}/requirements.txt"

python - <<'PY'
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit(
        "Errore: PyTorch non rileva CUDA. Controllare driver NVIDIA e "
        "installazione della build CUDA di PyTorch."
    )
PY

echo "Installazione completata. Avvia il server con: ./go.sh"
