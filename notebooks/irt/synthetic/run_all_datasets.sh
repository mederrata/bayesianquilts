#!/bin/bash
# Run remaining datasets sequentially for the synthetic evaluation pipeline.
# Uses sampled N(0,1) abilities as ground truth.
# GRIT already completed in results_v3.

set -e
cd /home/josh/workspace/bayesianquilts
source env/bin/activate
cd notebooks/irt/synthetic

OUTPUT_DIR="./results_v3"
COMMON_ARGS="--epochs 2000 --patience 50 --batch-size 256 --lr-decay-factor 0.9 --clip-norm 1.0"

echo "=============================================="
echo "Starting sequential dataset pipeline (sampled abilities)"
echo "GRIT already done, starting from RWA"
echo "Time: $(date)"
echo "=============================================="

# 1. RWA - reload NeuralGRM
echo ""
echo "=============================================="
echo "1/5: RWA (K=9, 9881 people, 22 items) [reload NeuralGRM]"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset rwa --output-dir "$OUTPUT_DIR" \
    --lr 5e-5 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/rwa_log.txt"

# 2. TMA - reload NeuralGRM
echo ""
echo "=============================================="
echo "2/5: TMA (K=2, 5410 people, 50 items) [reload NeuralGRM]"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset tma --output-dir "$OUTPUT_DIR" \
    --lr 2e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/tma_log.txt"

# 3. NPI - reload NeuralGRM
echo ""
echo "=============================================="
echo "3/5: NPI (K=2, 11243 people, 40 items) [reload NeuralGRM]"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset npi --output-dir "$OUTPUT_DIR" \
    --lr 2e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/npi_log.txt"

# 4. WPI - reload NeuralGRM
echo ""
echo "=============================================="
echo "4/5: WPI (K=2, 6019 people, 116 items) [reload NeuralGRM]"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset wpi --output-dir "$OUTPUT_DIR" \
    --lr 1e-4 --reload-neural-grm $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/wpi_log.txt"

# 5. EQSQ - full training (only 12 epochs in v2)
echo ""
echo "=============================================="
echo "5/5: EQSQ (K=4, 13256 people, 120 items) [full training]"
echo "Time: $(date)"
echo "=============================================="
python evaluate_synthetic.py --dataset eqsq --output-dir "$OUTPUT_DIR" \
    --lr 1e-4 $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/eqsq_log.txt"

echo ""
echo "=============================================="
echo "ALL DATASETS COMPLETE"
echo "Time: $(date)"
echo "=============================================="

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
