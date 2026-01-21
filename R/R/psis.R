#' Pareto Smoothed Importance Sampling (Wrapper around loo)
#'
#' @param log_ratios Numeric matrix of log importance ratios (Draws x Observations).
#' @return A list with elements:
#'   \item{log_weights}{Matrix of smoothed log weights}
#'   \item{weights}{Matrix of smoothed normalized weights}
#'   \item{khat}{Vector of Pareto k diagnostics}
#' @export
psislw <- function(log_ratios) {
  # Ensure matrix
  if (is.null(dim(log_ratios))) {
    log_ratios <- matrix(log_ratios, ncol = 1)
  }
  
  # Run PSIS from loo package
  # r_eff=NA assumes independent draws (MCMC efficiency not considered here for simple IS)
  psis_obj <- tryCatch({
    loo::psis(log_ratios, r_eff = NA)
  }, error = function(e) {
    warning("loo::psis failed, returning simple normalization: ", e$message)
    NULL
  })
  
  if (is.null(psis_obj)) {
    # Fallback equivalent to simple importance sampling
    max_log <- apply(log_ratios, 2, max)
    log_w <- t(t(log_ratios) - max_log) # Shift
    norm <- log(colSums(exp(log_w)))
    log_weights <- t(t(log_w) - norm)
    weights <- exp(log_weights)
    khat <- rep(Inf, ncol(log_ratios))
  } else {
    log_weights <- weights(psis_obj, log = TRUE)
    weights <- exp(log_weights)
    khat <- psis_obj$diagnostics$pareto_k
  }
  
  list(
    log_weights = log_weights,
    weights = weights,
    khat = khat
  )
}
