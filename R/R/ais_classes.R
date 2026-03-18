#' @import R6
NULL

#' Abstract Base Class for Likelihood Functions
#'
#' @description
#' Defines the interface for likelihood functions used in AIS.
#' Users must implement these methods for their specific models.
#'
#' @export
LikelihoodFunction <- R6::R6Class("LikelihoodFunction",
  public = list(
    #' @description Compute log-likelihood
    #' @param data Data object (list or environment)
    #' @param params List of parameters (S x K matrices or arrays)
    #' @return Matrix of log-likelihoods (S x N)
    log_likelihood = function(data, params) stop("Method log_likelihood not implemented"),

    #' @description Compute gradient of log-likelihood
    #' @param data Data object
    #' @param params List of parameters
    #' @return List of gradients (matching params structure, e.g. S x N x K)
    log_likelihood_gradient = function(data, params) stop("Method log_likelihood_gradient not implemented"),

    #' @description Compute diagonal of Hessian of log-likelihood
    #' @param data Data object
    #' @param params List of parameters
    #' @return List of hessian diagonals (matching params structure, e.g. S x N x K)
    log_likelihood_hessian_diag = function(data, params) stop("Method log_likelihood_hessian_diag not implemented"),

    #' @description Compute total log-likelihood across ALL observations for per-obs params.
    #'
    #' For per-observation transformed parameters with shape (S, N_params, K),
    #' computes for each (s, i):
    #'     result[s, i] = sum_j log l(d_j | params[s, i, :])
    #'
    #' Default implementation loops over the N dimension.
    #'
    #' @param data Data object
    #' @param params Per-observation transformed parameters
    #' @return Matrix of shape (S, N_params) with total log-likelihoods
    total_log_likelihood = function(data, params) {
      # Check if params have per-observation dimension (3D arrays)
      first_val <- params[[1]]
      if (length(dim(first_val)) <= 2) {
        # Global params (S, K): just sum the standard log_likelihood
        ll <- self$log_likelihood(data, params) # S x N
        return(matrix(rowSums(ll), ncol = 1))
      }

      # Per-observation params (S, N, K...) - loop over N dimension
      S <- dim(first_val)[1]
      N <- dim(first_val)[2]

      total_ll <- matrix(0, nrow = S, ncol = N)
      for (i in seq_len(N)) {
        p_i <- list()
        for (k in names(params)) {
          val <- params[[k]]
          if (length(dim(val)) == 3) {
            p_i[[k]] <- val[, i, , drop = FALSE]
            dim(p_i[[k]]) <- c(S, dim(val)[3])
          } else if (length(dim(val)) == 2) {
            p_i[[k]] <- val[, i]
          }
        }
        ll_full <- self$log_likelihood(data, p_i) # S x N_data
        total_ll[, i] <- rowSums(ll_full)
      }
      return(total_ll)
    }
  )
)

#' Abstract Base Class for AIS Transformations
#'
#' @export
Transformation <- R6::R6Class("Transformation",
  public = list(
    likelihood_fn = NULL,

    #' @description Initialize transformation
    #' @param likelihood_fn LikelihoodFunction object
    initialize = function(likelihood_fn) {
      self$likelihood_fn <- likelihood_fn
    },

    #' @description Apply transformation
    #' @param max_iter Integer, maximum iterations
    #' @param params List of parameters
    #' @param theta Top-level parameters (usually same as params)
    #' @param data Data object
    #' @param log_ell Matrix of log-likelihoods
    #' @param ... Additional arguments
    call = function(max_iter, params, theta, data, log_ell, ...) {
      stop("Method call not implemented")
    },

    #' @description Compute weighted and unweighted moments for parameters.
    #' @param params List of parameters where each value has shape (S, ...).
    #' @param weights Importance weights with shape (S, N).
    #' @return Named list mapping parameter names to lists with keys:
    #'   mean (unweighted), mean_w (weighted, N x ...), var, var_w.
    compute_moments = function(params, weights) {
      # Normalize weights column-wise (per N)
      w_sum <- colSums(weights)
      w_norm <- t(t(weights) / (w_sum + 1e-10)) # S x N

      moments <- list()

      for (name in names(params)) {
        val <- params[[name]] # S x K or S

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          # Scalar param: S vector
          mean_u <- mean(val)

          # Weighted mean: colSums(w * val) -> N
          mean_w <- colSums(w_norm * val)

          # Variance
          var_u <- mean((val - mean_u)^2)

          # Weighted variance: E[x^2] - E[x]^2
          val_mat <- matrix(val, nrow = length(val), ncol = ncol(weights))
          mean_sq_w <- colSums(w_norm * val_mat^2)
          var_w <- pmax(mean_sq_w - mean_w^2, 0)

          moments[[name]] <- list(mean = mean_u, mean_w = mean_w,
                                  var = var_u, var_w = var_w)

        } else {
          # Matrix param: S x K
          mean_u <- colMeans(val) # K
          mean_w <- t(w_norm) %*% val # N x K

          mean_sq_w <- t(w_norm) %*% (val^2) # N x K
          var_w <- pmax(mean_sq_w - mean_w^2, 0)

          var_u <- apply(val, 2, var) # K

          moments[[name]] <- list(mean = mean_u, mean_w = mean_w,
                                  var = var_u, var_w = var_w)
        }
      }
      return(moments)
    },

    #' @description Compute importance weights for transformed parameters.
    #'
    #' Implements the importance weight from the manuscript:
    #'   log eta = -log l(phi|d_i) + log|J| + log[pi(phi|D)/pi(theta|D)]
    #'
    #' For the variational case, uses surrogate density ratio.
    #' For the MCMC case, computes the full posterior LL ratio.
    #'
    #' @param likelihood_fn LikelihoodFunction object
    #' @param data Data object
    #' @param params_original Original parameters
    #' @param params_transformed Transformed parameters
    #' @param log_jacobian Log Jacobian determinant (S x N)
    #' @param variational Whether using variational approximation
    #' @param log_pi Log posterior/surrogate probability (S)
    #' @param log_ell_original Original log-likelihoods (S x N), optional
    #' @param surrogate_log_prob_fn Surrogate log probability function, optional
    #' @return List with eta_weights, psis_weights, khat, log_ell_new
    compute_importance_weights = function(likelihood_fn, data,
                                          params_original, params_transformed,
                                          log_jacobian,
                                          variational = FALSE,
                                          log_pi = NULL,
                                          log_ell_original = NULL,
                                          surrogate_log_prob_fn = NULL) {
      # Check if transformed params are per-observation (3D arrays)
      first_val <- params_transformed[[1]]
      is_per_obs <- (length(dim(first_val)) == 3)

      if (is_per_obs) {
        # Per-observation params: extract params for each obs i, evaluate LL,
        # and take only the diagonal log_ell_new[s, i] for obs i.
        S <- dim(first_val)[1]
        N <- dim(first_val)[2]
        log_ell_new <- matrix(0, nrow = S, ncol = N)
        for (i in seq_len(N)) {
          p_i <- list()
          for (k in names(params_transformed)) {
            val <- params_transformed[[k]]
            if (length(dim(val)) == 3) {
              p_i[[k]] <- val[, i, , drop = FALSE]
              dim(p_i[[k]]) <- c(S, dim(val)[3])
            } else {
              p_i[[k]] <- val[, i]
            }
          }
          ll_full <- likelihood_fn$log_likelihood(data, p_i) # S x N_data
          log_ell_new[, i] <- ll_full[, i]
        }
      } else {
        log_ell_new <- likelihood_fn$log_likelihood(data, params_transformed)
      }
      log_loo <- -log_ell_new

      if (variational && !is.null(surrogate_log_prob_fn)) {
        # Variational case: use surrogate density ratio
        log_pi_trans <- surrogate_log_prob_fn(params_transformed)
        if (is.null(dim(log_pi_trans))) {
          log_pi_trans <- matrix(log_pi_trans, ncol = 1)
        }
        if (is.null(dim(log_pi))) {
          log_pi_mat <- matrix(log_pi, ncol = 1)
        } else {
          log_pi_mat <- log_pi
        }
        delta_log_proposal <- log_pi_trans - log_pi_mat
      } else {
        # Non-variational case: approximate posterior ratio as 1
        # (standard PSIS-LOO assumption; exact ratio is O(SN^2) to compute)
        delta_log_proposal <- matrix(0, nrow = nrow(log_ell_new),
                                     ncol = ncol(log_ell_new))
      }

      # Full weight: LOO term + proposal correction + Jacobian
      log_eta_weights <- log_loo + delta_log_proposal + log_jacobian

      # Normalize
      eta_weights <- exp(log_eta_weights - apply(log_eta_weights, 2, max))
      eta_weights <- t(t(eta_weights) / colSums(eta_weights))

      # PSIS smoothing
      psis_res <- psislw(log_eta_weights - t(replicate(nrow(log_eta_weights),
                         apply(log_eta_weights, 2, max))))
      psis_weights <- psis_res$weights
      khat <- psis_res$khat

      list(eta_weights = eta_weights,
           psis_weights = psis_weights,
           khat = khat,
           log_ell_new = log_ell_new)
    }
  )
)
