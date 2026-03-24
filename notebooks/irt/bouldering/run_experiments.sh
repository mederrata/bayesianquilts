#!/bin/bash
# Run bouldering experiments for journal_article.tex
#
# Produces:
#   men/   - fitted models, results, forest plots for men
#   women/ - fitted models, results, forest plots for women
#   tex/   - LaTeX table fragments (bouldering_table1.tex, bouldering_table2.tex)
#
# Usage:
#   ./run_experiments.sh              # full run (real + synthetic, both genders)
#   ./run_experiments.sh --fast       # real data + plots only (skip synthetic)
#   ./run_experiments.sh --men-only   # just men

set -e
cd "$(dirname "$0")"

export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1

EXTRA_ARGS=""
if [[ "$1" == "--fast" ]]; then
    EXTRA_ARGS="--skip-synthetic"
    echo "Fast mode: skipping synthetic evaluation"
elif [[ "$1" == "--men-only" ]]; then
    EXTRA_ARGS="--gender men"
elif [[ "$1" == "--women-only" ]]; then
    EXTRA_ARGS="--gender women"
fi

echo "=============================================="
echo "Bouldering Experiments for Journal Article"
echo "Time: $(date)"
echo "=============================================="

uv run python run_bouldering_experiments.py \
    --gender both \
    --epochs 200 \
    --synthetic-epochs 500 \
    --lr 2e-4 \
    --batch-size 256 \
    --n-top 10 \
    $EXTRA_ARGS \
    2>&1 | tee experiments_log.txt

echo ""
echo "=============================================="
echo "COMPLETE - $(date)"
echo "=============================================="
echo ""
echo "Outputs:"
echo "  men/forest_top10_men.pdf"
echo "  men/ability_distributions_top10_men.pdf"
echo "  women/forest_top10_women.pdf"
echo "  women/ability_distributions_top10_women.pdf"
echo "  tex/bouldering_table1.tex"
echo "  tex/bouldering_table2.tex"
