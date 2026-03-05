#!/bin/bash
# Run all datasets sequentially for the synthetic evaluation pipeline.
# Uses sampled N(0,1) abilities as ground truth.
# 60% fully observed, 40% have 25% MCAR missingness.

set -e
cd /home/josh/workspace/bayesianquilts
source env/bin/activate
cd notebooks/irt/synthetic

OUTPUT_DIR="./results_v4"
COMMON_ARGS="--epochs 2000 --patience 50 --batch-size 256 --lr-decay-factor 0.9 --clip-norm 1.0 --missingness 0.25 --missing-respondent-frac 0.4"

echo "=============================================="
echo "Starting sequential dataset pipeline"
echo "Missingness: 25% MCAR for 40% of respondents (60% fully observed)"
echo "Time: $(date)"
echo "=============================================="

# Copy NeuralGRM models from v3 if they exist (avoid retraining)
# Skip RWA: v3 NeuralGRM had poor convergence (lr=5e-5, early stop at epoch 64)
for ds in grit tma npi; do
    src="./results_v3/${ds}/neural_grm"
    dst="./${OUTPUT_DIR}/${ds}/neural_grm"
    if [ -d "$src" ] && [ -f "${src}/params.h5" ]; then
        mkdir -p "$dst"
        cp -r "$src"/* "$dst"/
        echo "Copied NeuralGRM for ${ds} from v3"
    fi
done

# 1. GRIT
echo ""
echo "=============================================="
echo "1/6: GRIT (K=5, 4270 people, 12 items)"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset grit --output-dir "$OUTPUT_DIR" \
    --lr 2e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/grit_log.txt"

# 2. RWA
echo ""
echo "=============================================="
echo "2/6: RWA (K=9, 9881 people, 22 items)"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset rwa --output-dir "$OUTPUT_DIR" \
    --lr 2e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/rwa_log.txt"

# 3. TMA
echo ""
echo "=============================================="
echo "3/6: TMA (K=2, 5410 people, 50 items)"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset tma --output-dir "$OUTPUT_DIR" \
    --lr 2e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/tma_log.txt"

# 4. NPI
echo ""
echo "=============================================="
echo "4/6: NPI (K=2, 11243 people, 40 items)"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset npi --output-dir "$OUTPUT_DIR" \
    --lr 2e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/npi_log.txt"

# 5. WPI
echo ""
echo "=============================================="
echo "5/6: WPI (K=2, 6019 people, 116 items)"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset wpi --output-dir "$OUTPUT_DIR" \
    --lr 1e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/wpi_log.txt"

# 6. EQSQ
echo ""
echo "=============================================="
echo "6/6: EQSQ (K=4, 13256 people, 120 items)"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset eqsq --output-dir "$OUTPUT_DIR" \
    --lr 1e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/eqsq_log.txt"

echo ""
echo "=============================================="
echo "ALL DATASETS COMPLETE"
echo "Time: $(date)"
echo "=============================================="

# Run PSIS-LOO on all baseline models
echo ""
echo "=============================================="
echo "Running PSIS-LOO diagnostics on baseline models"
echo "Time: $(date)"
echo "=============================================="
python run_psis_loo.py --results-dir "$OUTPUT_DIR" --n-samples 100 2>&1 | tee "${OUTPUT_DIR}/psis_loo_log.txt"

# Print summary
echo ""
echo "Results summary:"
for ds in grit rwa tma npi wpi eqsq; do
    if [ -f "${OUTPUT_DIR}/${ds}/results.json" ]; then
        echo "  ${ds}: $(cat ${OUTPUT_DIR}/${ds}/results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
b = r['baseline']
i = r['imputed']
print(f\"Baseline Spearman={b['spearman_r']:.4f} RMSE={b['rmse']:.4f} | Imputed Spearman={i['spearman_r']:.4f} RMSE={i['rmse']:.4f}\")
")"
    else
        echo "  ${ds}: FAILED (no results.json)"
    fi
done
