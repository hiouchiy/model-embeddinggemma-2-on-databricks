#!/bin/bash
# Install the EmbeddingGemma 2 runtime on serverless GPU (environment version 4). torch 2.10 / cu128 replaces the
# preinstalled torch. Used by the notebooks and air_job/generate.yaml; the pins live in embeddinggemma2_core.py.
# The serving registration (notebooks/register_serving.py) packs the environment this leaves (env_pack).
#
#   bash scripts/install.sh                         # from the repo root (AIR job)
#   PYTHON=/path/to/python bash scripts/install.sh  # into a specific interpreter (the notebooks pass their own)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
export HF_HUB_ENABLE_HF_TRANSFER=0     # serverless GPU presets it to 1 but does not install hf_transfer
INDEX=$("$PY" -c "import embeddinggemma2_core as c; print(c.TORCH_INDEX)")
mapfile -t REQS < <("$PY" -c "import embeddinggemma2_core as c; print('\n'.join(c.RUNTIME_REQUIREMENTS))")
"$PY" -m pip install -q --no-cache-dir --extra-index-url "$INDEX" "${REQS[@]}" 2>&1 | tail -30
"$PY" -m pip check || true
"$PY" -c "import torch, torchvision, transformers, sentence_transformers as st; print('[install] torch', torch.__version__, 'torchvision', torchvision.__version__, 'transformers', transformers.__version__, 'sentence-transformers', st.__version__, 'cuda', torch.cuda.is_available())"
