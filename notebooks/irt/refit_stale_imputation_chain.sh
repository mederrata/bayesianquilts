#!/bin/bash
# Refit stacking model + all 3 MCMC variants for a dataset whose stacking was
# stale (pre-reorient). Backs up old artifacts with .stale_<date> suffix.
#
# Usage: ./refit_stale_imputation_chain.sh <dataset> <python_path> <log_dir>

set -e

DATASET="$1"
PYTHON="${2:-/home/josh/workspace/bayesianquilts/.venv-rocm/bin/python}"
LOGDIR="${3:-/home/josh/workspace/bayesianquilts/notebooks/irt/logs/refit}"

cd /home/josh/workspace/bayesianquilts/notebooks/irt
mkdir -p "$LOGDIR"

ts=$(date +%Y%m%d_%H%M%S)

# Backup stale artifacts (preserve mtime)
if [ -f "$DATASET/pairwise_stacking_model.yaml" ]; then
  cp -p "$DATASET/pairwise_stacking_model.yaml" "$DATASET/pairwise_stacking_model.stale_${ts}.yaml"
fi
for v in baseline pairwise mixed; do
  src="$DATASET/mcmc_samples/mcmc_${v}.npz"
  if [ -f "$src" ]; then
    cp -p "$src" "$DATASET/mcmc_samples/mcmc_${v}.stale_${ts}.npz"
  fi
done

echo "[$(date)] === Step 1: refit pairwise stacking ($DATASET) ==="
JAX_PLATFORMS=cpu "$PYTHON" fit_stacking_model.py --dataset "$DATASET" \
  > "$LOGDIR/${DATASET}_stacking_${ts}.log" 2>&1
echo "[$(date)] stacking refit done"

echo "[$(date)] === Step 2: rerun MCMC (baseline pairwise mixed) ==="
JAX_PLATFORMS=cpu "$PYTHON" run_marginal_mcmc.py --dataset "$DATASET" \
  --variants baseline pairwise mixed --num-chains 4 --num-warmup 3000 \
  --num-samples 500 --step-size 0.01 --seed 4252 \
  > "$LOGDIR/${DATASET}_mcmc_chain_${ts}.log" 2>&1
echo "[$(date)] MCMC rerun done"

echo "[$(date)] === Final artifact mtimes ==="
ls -la "$DATASET/pairwise_stacking_model.yaml" "$DATASET/mcmc_samples/"
