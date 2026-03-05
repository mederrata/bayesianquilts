#!/bin/bash
# Run all IRT GRM notebooks sequentially.
set -e
cd /home/josh/workspace/bayesianquilts
source env/bin/activate
cd notebooks/irt

for ds in grit rwa npi tma wpi eqsq; do
    echo ""
    echo "=============================================="
    echo "Running ${ds}/grm_single_scale.ipynb"
    echo "Time: $(date)"
    echo "=============================================="
    cd "$ds"
    jupyter nbconvert --to notebook --execute grm_single_scale.ipynb \
        --ExecutePreprocessor.timeout=3600 \
        --output grm_single_scale_executed.ipynb 2>&1 || echo "FAILED: $ds"
    cd ..
done

# RWA factorized notebook
echo ""
echo "=============================================="
echo "Running rwa/factorized_grm_missing.ipynb"
echo "Time: $(date)"
echo "=============================================="
cd rwa
jupyter nbconvert --to notebook --execute factorized_grm_missing.ipynb \
    --ExecutePreprocessor.timeout=3600 \
    --output factorized_grm_missing_executed.ipynb 2>&1 || echo "FAILED: rwa factorized"
cd ..

echo ""
echo "=============================================="
echo "ALL NOTEBOOKS COMPLETE"
echo "Time: $(date)"
echo "=============================================="
