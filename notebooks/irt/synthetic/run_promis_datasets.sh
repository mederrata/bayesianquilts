#!/bin/bash
# Run PROMIS datasets for the synthetic evaluation pipeline.
# Uses same pipeline as openpsychometrics but with PROMIS data loaders.

set -e
cd /home/josh/workspace/bayesianquilts/notebooks/irt/synthetic

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

OUTPUT_DIR="./results_promis"
COMMON_ARGS="--epochs 500 --patience 20 --batch-size 256 --lr-decay-factor 0.975 --clip-norm 1.0 --noisy-dim 2 --mcmc"

echo "=============================================="
echo "PROMIS Synthetic Evaluation Pipeline"
echo "Time: $(date)"
echo "=============================================="

mkdir -p "$OUTPUT_DIR"

# --- Single-scale PROMIS datasets ---

# 1. Sleep-Wake (K=5, ~126 items)
echo -e "\n=== Sleep-Wake ==="
uv run python evaluate_synthetic.py --dataset promis_sleep --output-dir "$OUTPUT_DIR" \
    --lr 5e-4 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
    $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/promis_sleep_log.txt"

# 2. Substance Use (K=5, ~144 items)
echo -e "\n=== Substance Use ==="
uv run python evaluate_synthetic.py --dataset promis_substance_use --output-dir "$OUTPUT_DIR" \
    --lr 5e-4 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
    $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/promis_substance_use_log.txt"

# --- COPD per-domain datasets (K=5) ---

for domain in depression anxiety anger social_satisfaction fatigue_experience fatigue_impact pain_interference pain_behavior physical_function; do
    echo -e "\n=== COPD ${domain} ==="
    uv run python evaluate_synthetic.py --dataset "copd_${domain}" --output-dir "$OUTPUT_DIR" \
        --lr 5e-4 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
        $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/copd_${domain}_log.txt"
done

# --- Neuropathic Pain per-domain datasets (K=5) ---

for domain in pain_interference pain_behavior global_health physical_function; do
    echo -e "\n=== NP ${domain} ==="
    uv run python evaluate_synthetic.py --dataset "np_${domain}" --output-dir "$OUTPUT_DIR" \
        --lr 5e-4 --discrimination-prior half_normal --discrimination-prior-scale 2.0 \
        $COMMON_ARGS 2>&1 | tee "${OUTPUT_DIR}/np_${domain}_log.txt"
done

echo -e "\n=============================================="
echo "ALL PROMIS DATASETS COMPLETE - $(date)"
echo "=============================================="

# Print summary
for ds in promis_sleep promis_substance_use \
    copd_depression copd_anxiety copd_anger copd_social_satisfaction \
    copd_fatigue_experience copd_fatigue_impact \
    copd_pain_interference copd_pain_behavior copd_physical_function \
    np_pain_interference np_pain_behavior np_global_health np_physical_function; do
    if [ -f "${OUTPUT_DIR}/${ds}/results.json" ]; then
        echo "  ${ds}: $(cat ${OUTPUT_DIR}/${ds}/results.json | python3 -c "
import sys, json
r = json.load(sys.stdin)
b = r['baseline']
m = r['mice_only']
i = r['imputed']
print(f'Base rho={b[\"spearman_r\"]:.4f} | MICE rho={m[\"spearman_r\"]:.4f} | Mixed rho={i[\"spearman_r\"]:.4f}')
")"
    else
        echo "  ${ds}: FAILED"
    fi
done
