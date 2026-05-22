#' bayesianquilts: Adaptive Importance Sampling and Quilted-Model Scaling
#'
#' Tools for interpretable Bayesian machine learning built around two ideas:
#'
#' * Adaptive Importance Sampling (AIS) for Leave-One-Out (LOO)
#'   cross-validation, with small-step (LL/NLL/KL/NKL/Variance/PMM1-3) and
#'   global (MM1-3/MixIS) transformations targeting finite-variance IS
#'   weights when standard PSIS-LOO fails.
#'
#' * The additive interaction decomposition of multi-way regression
#'   coefficients into orthogonal components, with the
#'   generalization-preserving prior scales of Chang (2026) derived from
#'   each component's effective sample size.
#'
#' See `system.file("doc", "ais_loo.Rmd", package = "bayesianquilts")`
#' for a walkthrough of the AIS workflow.
#'
#' @keywords internal
#' @aliases bayesianquilts-package
#' @import R6
#' @importFrom stats sd var family setNames cov as.formula
#' @importFrom loo psis
"_PACKAGE"
