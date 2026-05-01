#!/usr/bin/env bash
# Track B launch staging — run after smoke test passes and code is committed.
# Each block is a complete command; uncomment the ones you want to launch.
# Designed to be run from ~/workspace/bayesianquilts/notebooks/irt on either
# miniryzen or tankieryzen. Non-interactive ssh must use /home/josh/.local/bin/uv.

set -euo pipefail

UV=/home/josh/.local/bin/uv  # full path for non-interactive ssh safety
IRT_DIR=~/workspace/bayesianquilts/notebooks/irt
LOGS=$IRT_DIR/logs

cd "$IRT_DIR"
mkdir -p "$LOGS"

# Common flags for every Track B NUTS run:
#   --sampler nuts        use NUTS (mu-drop prior is unconditional in code)
#   --dense-mass          full covariance mass matrix
#   --num-chains 4        4 chains for R-hat diagnostics
#   --num-warmup 2000     longer warmup helps the new dense-mass adapt
#   --num-samples 1000    1000 post-warmup samples per chain
#   --step-size 5e-4      new project default
COMMON_NUTS="--sampler nuts --dense-mass --num-chains 4 --num-warmup 2000 --num-samples 1000 --step-size 5e-4"

# ---------------------------------------------------------------------------
# GROUP 1: baseline-init pairwise/mixed for datasets whose baseline converged
# ---------------------------------------------------------------------------

# rwa: baseline converged (R-hat 1.034); pairwise + mixed both FAILED (R-hat 9.4/10)
nohup $UV run python run_marginal_mcmc.py --dataset rwa --variants pairwise mixed \
  --model-dir $IRT_DIR/rwa/grm_baseline_bposterior \
  $COMMON_NUTS > $LOGS/track_b_rwa.log 2>&1 & echo "rwa: $!"

# wpi: baseline converged (R-hat 1.033); pairwise FAILED (R-hat 9.9); mixed already OK
nohup $UV run python run_marginal_mcmc.py --dataset wpi --variants pairwise \
  --model-dir $IRT_DIR/wpi/grm_baseline_bposterior \
  $COMMON_NUTS > $LOGS/track_b_wpi.log 2>&1 & echo "wpi: $!"

# promis_sleep: baseline converged (R-hat 1.069); pairwise + mixed not yet fit
nohup $UV run python run_marginal_mcmc.py --dataset promis_sleep --variants pairwise mixed \
  --model-dir $IRT_DIR/promis_sleep/grm_baseline_bposterior \
  $COMMON_NUTS > $LOGS/track_b_promis_sleep.log 2>&1 & echo "promis_sleep: $!"

# grit: baseline + pairwise OK; mixed marginal (R-hat 1.099). Optional re-fit.
# nohup $UV run python run_marginal_mcmc.py --dataset grit --variants mixed \
#   --model-dir $IRT_DIR/grit/grm_baseline_bposterior \
#   $COMMON_NUTS > $LOGS/track_b_grit.log 2>&1 & echo "grit: $!"

# ---------------------------------------------------------------------------
# GROUP 2: failed baselines — re-fit from scratch with mu-drop + dense-mass
# Use default grm_baseline dir; mu-drop is automatic in the new code.
# These MUST converge before their pairwise/mixed variants can run (Group 3).
# ---------------------------------------------------------------------------

# npi: all attempts failed (R-hat 4+ up to 130)
nohup $UV run python run_marginal_mcmc.py --dataset npi --variants baseline \
  $COMMON_NUTS > $LOGS/track_b_npi_baseline.log 2>&1 & echo "npi baseline: $!"

# promis_copd: MALA baseline failed (R-hat 13)
nohup $UV run python run_marginal_mcmc.py --dataset promis_copd --variants baseline \
  $COMMON_NUTS > $LOGS/track_b_promis_copd_baseline.log 2>&1 & echo "promis_copd baseline: $!"

# promis_neuropathic_pain: MALA baseline failed (R-hat 11)
nohup $UV run python run_marginal_mcmc.py --dataset promis_neuropathic_pain --variants baseline \
  $COMMON_NUTS > $LOGS/track_b_promis_np_baseline.log 2>&1 & echo "promis_np baseline: $!"

# promis_substance_use: MALA baseline marginal (R-hat 2.91 on ddifficulties). Optional redo.
# nohup $UV run python run_marginal_mcmc.py --dataset promis_substance_use --variants baseline \
#   $COMMON_NUTS > $LOGS/track_b_promis_substance_baseline.log 2>&1 & echo "promis_substance baseline: $!"

# ---------------------------------------------------------------------------
# GROUP 3: pairwise/mixed for previously-failed baselines — launches AFTER
#          Group 2 baselines converge. Needs a fresh bposterior dir built from
#          the new baseline NPZ first. Uncomment and run the init_from_baseline
#          step, then the MCMC launch.
# ---------------------------------------------------------------------------

# # After npi baseline converges:
# $UV run python init_from_baseline_posterior.py --dataset npi \
#   --baseline-npz npi/mcmc_samples/mcmc_baseline.npz \
#   --output-dir npi/grm_baseline_bposterior
# nohup $UV run python run_marginal_mcmc.py --dataset npi --variants pairwise mixed \
#   --model-dir $IRT_DIR/npi/grm_baseline_bposterior \
#   $COMMON_NUTS > $LOGS/track_b_npi_pairmix.log 2>&1 &

# # After promis_copd baseline converges:
# $UV run python init_from_baseline_posterior.py --dataset promis_copd \
#   --baseline-npz promis_copd/mcmc_samples/mcmc_baseline.npz \
#   --output-dir promis_copd/grm_baseline_bposterior
# nohup $UV run python run_marginal_mcmc.py --dataset promis_copd --variants pairwise mixed \
#   --model-dir $IRT_DIR/promis_copd/grm_baseline_bposterior \
#   $COMMON_NUTS > $LOGS/track_b_promis_copd_pairmix.log 2>&1 &

# # After promis_neuropathic_pain baseline converges:
# $UV run python init_from_baseline_posterior.py --dataset promis_neuropathic_pain \
#   --baseline-npz promis_neuropathic_pain/mcmc_samples/mcmc_baseline.npz \
#   --output-dir promis_neuropathic_pain/grm_baseline_bposterior
# nohup $UV run python run_marginal_mcmc.py --dataset promis_neuropathic_pain --variants pairwise mixed \
#   --model-dir $IRT_DIR/promis_neuropathic_pain/grm_baseline_bposterior \
#   $COMMON_NUTS > $LOGS/track_b_promis_np_pairmix.log 2>&1 &

wait
echo "All launched Track B runs completed (or wait returned)."
