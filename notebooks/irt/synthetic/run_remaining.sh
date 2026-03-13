#!/bin/bash
set -e
cd /home/josh/workspace/bayesianquilts/notebooks/irt/synthetic

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

OUTPUT_DIR="./results"
COMMON_ARGS="--epochs 500 --patience 20 --batch-size 256 --lr-decay-factor 0.975 --clip-norm 1.0 --elpd-loo --noisy-dim 2"

# 3. TMA (K=2, 50 items)
echo -e "\n=== 3/6: TMA ==="
uv run python evaluate_synthetic.py --dataset tma --output-dir "$OUTPUT_DIR" \
    --lr 1e-3 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
    $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/tma_log.txt"

# 4. NPI (K=2, 40 items)
echo -e "\n=== 4/6: NPI ==="
uv run python evaluate_synthetic.py --dataset npi --output-dir "$OUTPUT_DIR" \
    --lr 1e-3 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
    $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/npi_log.txt"

# 5. WPI (K=2, 116 items)
echo -e "\n=== 5/6: WPI ==="
uv run python evaluate_synthetic.py --dataset wpi --output-dir "$OUTPUT_DIR" \
    --lr 5e-4 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
    $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/wpi_log.txt"

# 6. EQSQ (K=4, 120 items)
echo -e "\n=== 6/6: EQSQ ==="
uv run python evaluate_synthetic.py --dataset eqsq --output-dir "$OUTPUT_DIR" \
    --lr 5e-4 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
    $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/eqsq_log.txt"

echo -e "\n=============================================="
echo "REMAINING DATASETS COMPLETE - $(date)"
echo "=============================================="
