#!/bin/bash
# Run all AIS experiments inside rocm/jax-training Docker container
# Usage: bash run_all_docker.sh

set -e

DOCKER_IMAGE="rocm/jax-training:latest"
WORKSPACE="/home/josh/workspace/bayesianquilts"
RESULTS_DIR="${WORKSPACE}/results"
mkdir -p "${RESULTS_DIR}"

# Common docker args
DOCKER_ARGS=(
    --rm
    --device=/dev/kfd
    --device=/dev/dri
    --group-add video
    -v "${WORKSPACE}:/workspace"
    -v "/tmp:/host_tmp"
    -w /workspace
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0
)

# Install deps and run a script
run_experiment() {
    local script="$1"
    local logfile="$2"
    echo "=== Starting $(basename $script) -> $logfile ==="
    docker run "${DOCKER_ARGS[@]}" "${DOCKER_IMAGE}" bash -c "
        pip install -q tfp-nightly arviz 'optax>=0.2.4' 'flax>=0.12.4' orbax-checkpoint grain tqdm polars blackjax scikit-learn 2>/dev/null
        export PYTHONPATH=/workspace/python:\$PYTHONPATH
        python3 -u /host_tmp/$(basename $script)
    " > "${logfile}" 2>&1
    echo "=== Finished $(basename $script) ==="
}

echo "Starting all experiments at $(date)"
echo "Docker image: ${DOCKER_IMAGE}"
echo ""

# Run experiments sequentially (GPU memory shared)
# run_experiment /tmp/rerun_roaches.py "${RESULTS_DIR}/roaches.log"  # already done
run_experiment /tmp/rerun_ovarian_lr.py "${RESULTS_DIR}/ovarian_lr.log"
run_experiment /tmp/rerun_ovarian_nn.py "${RESULTS_DIR}/ovarian_nn.log"
run_experiment /tmp/rerun_gbsg2_npes.py "${RESULTS_DIR}/gbsg2_npes.log"
run_experiment /tmp/rerun_gbsg2_qpes.py "${RESULTS_DIR}/gbsg2_qpes.log"

echo ""
echo "All experiments finished at $(date)"
