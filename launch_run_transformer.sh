#!/usr/bin/env bash
# End-to-end helper for the transformer CTC prototype.
# Usage: ./launch_run_transformer.sh [run_name]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH=src
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

RUN_NAME="${1:-transformer_small_$(date +%Y%m%d_%H%M%S)}"
GEN_CFG="config/generation.yaml"
TRAIN_CFG="config/transformer_small.yaml"
RUN_DIR="outputs/${RUN_NAME}"

if [ "${FORCE_REGEN:-0}" = "1" ] || [ ! -f data/datasets/simple_baseline/manifests/train.jsonl ]; then
  echo "==> Generating dataset via ${GEN_CFG}"
  python3 -m audio2morse.data.generate_synthetic_morse --config "${GEN_CFG}"
else
  echo "==> Skipping data generation (manifests found). Set FORCE_REGEN=1 to regenerate."
fi

echo "==> Training run ${RUN_NAME} with ${TRAIN_CFG}"
python3 -m audio2morse.training.train --config "${TRAIN_CFG}" --run-name "${RUN_NAME}"

CKPT="${RUN_DIR}/best.pt"
echo "==> Adding predictions to manifests using ${CKPT}"
python3 analyses/add_predictions.py \
  --checkpoint "${CKPT}" \
  --config "${TRAIN_CFG}" \
  --train data/datasets/simple_baseline/manifests/train.jsonl \
  --val   data/datasets/simple_baseline/manifests/val.jsonl \
  --test  data/datasets/simple_baseline/manifests/test.jsonl \
  --run-dir "${RUN_DIR}"

echo "==> Running R analysis"
Rscript analyses/analyze_preds.R --run-dir "${RUN_DIR}"

echo "Done. Outputs in ${RUN_DIR}"
