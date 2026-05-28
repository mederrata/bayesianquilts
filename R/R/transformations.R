#' @include ais_classes.R
NULL

#' Small Step Transformation Base Class
#'
#' Transformations of the form T(theta) = theta + h * Q(theta).
#'
#' @export
SmallStepTransformation <- R6::R6Class("SmallStepTransformation",
  inherit = Transformation,
  public = list(

    #' @description Normalize vector field Q
    #' @param Q List of gradient/step arrays
    #' @param theta_std List of standard deviations
    normalize_vector_field = function(Q, theta_std = NULL) {
      Q_keys <- names(Q)

      # Standardize
      if (!is.null(theta_std)) {
        for (k in Q_keys) {
          val <- Q[[k]]
          std <- theta_std[[k]]
          if (!is.null(std)) {
            if (length(dim(val)) == 3) { # S x N x K
              Q[[k]] <- sweep(val, 3, std + 1e-6, "/")
            } else if (length(dim(val)) == 2) { # S x N
              Q[[k]] <- val / (std + 1e-6)
            }
          }
        }
      }

      # Compute max magnitude per (S, N) across all params
      max_mags <- list()
      for (k in Q_keys) {
        val <- Q[[k]]
        if (length(dim(val)) == 3) {
          max_mags[[k]] <- apply(abs(val), c(1, 2), max)
        } else {
          max_mags[[k]] <- abs(val) # S x N
        }
      }

      total_max <- max_mags[[1]]
      if (length(max_mags) > 1) {
        for (i in 2:length(max_mags)) {
          total_max <- pmax(total_max, max_mags[[i]])
        }
      }

      # Max over S -> N
      Q_norm_max <- apply(total_max, 2, max)

      list(Q = Q, Q_norm_max = Q_norm_max)
    },

    #' @description Apply transformation step.
    #' @param max_iter Integer; ignored for the single-step transforms in
    #'   this file but retained for API compatibility.
    #' @param params Original posterior samples (named list of `(S, ...)` arrays).
    #' @param theta Per-observation expansion of `params`, shape `(S, N, ...)`.
    #' @param data Data object.
    #' @param log_ell Matrix of log-likelihoods, shape `(S, N)`.
    #' @param hbar Step size for `T(theta) = theta + h * Q(theta)`.
    #' @param theta_std Per-parameter standard deviations used by the
    #'   normalization step. If `NULL`, no standardization.
    #' @param log_ell_original Original `(S, N)` log-likelihoods (used by
    #'   transforms that compute moments under leave-one-out weights).
    #' @param log_pi Posterior or surrogate log-probability per sample.
    #' @param variational If `TRUE`, use the surrogate density ratio in the
    #'   IS-weight computation.
    #' @param surrogate_log_prob_fn Optional function returning log-surrogate
    #'   density of transformed samples (variational case only).
    #' @param ... Forwarded to `compute_Q`.
    call = function(max_iter, params, theta, data, log_ell, hbar = 1.0,
                    theta_std = NULL, log_ell_original = NULL,
                    log_pi = NULL, variational = FALSE,
                    surrogate_log_prob_fn = NULL, ...) {

      # 1. Compute Q
      Q <- self$compute_Q(theta, data, params, log_ell,
                          log_ell_original = log_ell_original,
                          log_pi = log_pi, ...)

      # 2. Normalize
      norm_res <- self$normalize_vector_field(Q, theta_std)
      Q <- norm_res$Q
      Q_norm_max <- norm_res$Q_norm_max

      # h = hbar / norm
      h <- hbar / (Q_norm_max + 1e-8) # Vector N

      # 3. Update theta
      S <- nrow(log_ell)
      N <- ncol(log_ell)
      theta_new <- list()
      for (k in names(theta)) {
        t_val <- theta[[k]]
        q_val <- Q[[k]]

        if (length(dim(t_val)) == 3) {
          # S x N x K (or S x 1 x K)
          update <- sweep(q_val, 2, h, "*")
          theta_new[[k]] <- t_val + update
        } else {
          # S x N
          update <- sweep(q_val, 2, h, "*")
          theta_new[[k]] <- t_val + update
        }
      }

      # 4. Jacobian
      div_Q <- self$compute_divergence_Q(theta, data, params, log_ell, ...)

      if (!is.null(dim(div_Q)) && length(dim(div_Q)) == 2) {
        term <- 1.0 + sweep(div_Q, 2, h, "*")
        log_jacobian <- log(abs(term))
      } else {
        log_jacobian <- matrix(0, nrow = S, ncol = N)
      }

      # 5. Compute importance weights
      iw <- self$compute_importance_weights(
        self$likelihood_fn, data, params, theta_new,
        log_jacobian, variational, log_pi, log_ell_original,
        surrogate_log_prob_fn
      )

      log_ell_new <- iw$log_ell_new
      exp_log_ell_new <- exp(log_ell_new)

      # Compute LOO metrics
      ll_loo_eta <- colSums(iw$eta_weights * exp_log_ell_new)
      ll_loo_psis <- colSums(iw$psis_weights * exp_log_ell_new)
      p_loo_eta <- colSums(iw$eta_weights * exp(log_ell_new))
      p_loo_psis <- colSums(iw$psis_weights * exp(log_ell_new))

      list(
        theta_new = theta_new,
        log_jacobian = log_jacobian,
        eta_weights = iw$eta_weights,
        psis_weights = iw$psis_weights,
        khat = iw$khat,
        log_ell_new = log_ell_new,
        weight_entropy = entropy(iw$eta_weights),
        psis_entropy = entropy(iw$psis_weights),
        p_loo_eta = p_loo_eta,
        p_loo_psis = p_loo_psis,
        ll_loo_eta = ll_loo_eta,
        ll_loo_psis = ll_loo_psis
      )
    },

    #' @description Compute the vector field Q
    compute_Q = function(theta, data, params, current_log_ell, ...) {
      stop("compute_Q not implemented")
    },

    #' @description Compute divergence of Q for Jacobian approximation.
    #'
    #' Uses numDeriv for numerical trace(dQ/dtheta) computation.
    #' For T(theta) = theta + h*Q(theta), log|J| ~ log|1 + h*div(Q)|.
    #'
    #' @param theta Current parameter PyTree (list of S x N x K arrays)
    #' @param data Data object
    #' @param params Original parameters (S x K)
    #' @param current_log_ell Current log-likelihoods (S x N)
    #' @param ... Extra args forwarded to compute_Q
    #' @return Matrix (S x N) of divergence values
    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      if (!requireNamespace("numDeriv", quietly = TRUE)) {
        # Fallback to zero divergence if numDeriv not available
        return(matrix(0, nrow = nrow(current_log_ell),
                      ncol = ncol(current_log_ell)))
      }

      S <- nrow(current_log_ell)
      N <- ncol(current_log_ell)

      # Flatten theta to a vector per (s, n) and compute tr(dQ/dtheta)
      # For efficiency, compute divergence for a representative sample
      # then broadcast, or compute per (s, n) pair.
      #
      # Strategy: for each observation n, pick the first sample s=1,
      # flatten theta[1, n, :] across all params to a vector,
      # define Q as a function of that vector, compute numerical Jacobian,
      # and take the trace. Then broadcast across S (Q doesn't depend on s
      # for most transforms, so div(Q) is approximately constant across S).

      div_Q <- matrix(0, nrow = S, ncol = N)

      # Build flattening/unflattening helpers
      param_names <- names(theta)
      param_sizes <- integer(length(param_names))
      for (idx in seq_along(param_names)) {
        val <- theta[[param_names[idx]]]
        if (length(dim(val)) == 3) {
          param_sizes[idx] <- dim(val)[3]
        } else {
          param_sizes[idx] <- 1L
        }
      }
      K_total <- sum(param_sizes)

      # For small K_total, compute full numerical divergence
      # For large K_total, fall back to zero
      if (K_total > 50) {
        return(div_Q)
      }

      # Compute divergence for each n using s=1 as representative
      extra_args <- list(...)
      for (n in seq_len(N)) {
        # Extract theta[1, n, :] as flat vector
        flat_theta <- numeric(K_total)
        pos <- 1L
        for (idx in seq_along(param_names)) {
          val <- theta[[param_names[idx]]]
          k_i <- param_sizes[idx]
          if (length(dim(val)) == 3) {
            flat_theta[pos:(pos + k_i - 1)] <- val[1, n, ]
          } else {
            flat_theta[pos] <- val[1, n]
          }
          pos <- pos + k_i
        }

        # Define Q as function of flat theta vector for this (s=1, n)
        Q_flat_fn <- function(fv) {
          # Unflatten fv into theta structure for (s=1, n)
          theta_local <- theta
          pos <- 1L
          for (idx in seq_along(param_names)) {
            k_i <- param_sizes[idx]
            if (length(dim(theta_local[[param_names[idx]]])) == 3) {
              theta_local[[param_names[idx]]][1, n, ] <- fv[pos:(pos + k_i - 1)]
            } else {
              theta_local[[param_names[idx]]][1, n] <- fv[pos]
            }
            pos <- pos + k_i
          }
          # Compute Q at modified theta
          Q_val <- do.call(self$compute_Q,
                           c(list(theta_local, data, params, current_log_ell),
                             extra_args))
          # Extract Q[1, n, :] as flat vector
          out <- numeric(K_total)
          pos <- 1L
          for (idx in seq_along(param_names)) {
            k_i <- param_sizes[idx]
            qv <- Q_val[[param_names[idx]]]
            if (length(dim(qv)) == 3) {
              out[pos:(pos + k_i - 1)] <- qv[1, n, ]
            } else {
              out[pos] <- qv[1, n]
            }
            pos <- pos + k_i
          }
          return(out)
        }

        # Compute Jacobian and take trace
        tryCatch({
          jac <- numDeriv::jacobian(Q_flat_fn, flat_theta)
          trace_val <- sum(diag(jac))
          # Broadcast across S
          div_Q[, n] <- trace_val
        }, error = function(e) {
          div_Q[, n] <<- 0
        })
      }

      return(div_Q)
    }
  )
)

#' Likelihood Descent Transformation
#'
#' Q = -grad(log_ell). Moves samples in the direction that decreases
#' the likelihood of the left-out observation.
#'
#' Uses the analytical divergence div(Q) = -tr(H) (Hessian diagonal sum),
#' matching the Python implementation. This avoids the zero-Jacobian bug
#' from numerical differencing when h*div(Q) is small.
#'
#' @export
LikelihoodDescent <- R6::R6Class("LikelihoodDescent",
  inherit = SmallStepTransformation,
  public = list(
    compute_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }
      # Q = -grad(log_ell)
      lapply(grad_ll, function(x) -x)
    },

    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_doubleprime)) {
        diag_hess <- kwargs$log_ell_doubleprime
      } else {
        diag_hess <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)
      }

      # div(Q) = -tr(H) summed across all parameter leaves
      total_div <- matrix(0, nrow = nrow(current_log_ell),
                          ncol = ncol(current_log_ell))
      for (k in names(diag_hess)) {
        val <- diag_hess[[k]]
        if (length(dim(val)) == 3) {
          total_div <- total_div - apply(val, c(1, 2), sum)
        } else {
          total_div <- total_div - val
        }
      }

      return(total_div)
    }
  )
)


#' Natural-gradient Likelihood Descent Transformation
#'
#' \code{Q_NLL = Sigma \%*\% (-grad log ell_i)}, where `Sigma` is the
#' posterior covariance. Approximates the influence function
#' \code{Jinv \%*\% (grad log ell_i)} without the `pi(theta | D)` prefactor of
#' the KL flow. Cheaper than KL (no posterior evaluation) and more directly
#' aligned with the optimal LOO shift.
#'
#' Analytical divergence: div(Q_NLL) = -tr(Sigma %*% H) with diagonal-Hessian
#' approximation: -sum_k Sigma_kk * H_kk.
#'
#' @export
NaturalLikelihoodDescent <- R6::R6Class("NaturalLikelihoodDescent",
  inherit = SmallStepTransformation,
  public = list(
    posterior_cov = NULL,

    #' @description Initialize
    #' @param likelihood_fn LikelihoodFunction object
    #' @param posterior_cov Posterior covariance matrix (K_total x K_total)
    initialize = function(likelihood_fn, posterior_cov) {
      super$initialize(likelihood_fn)
      self$posterior_cov <- posterior_cov
    },

    compute_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }

      # Q_LL = -grad log ell
      Q_ll <- lapply(grad_ll, function(x) -x)

      # Flatten to (S, N, K_total), precondition by Sigma, unflatten.
      flat <- flatten_pytree_leaves(Q_ll, current_log_ell)
      S <- nrow(current_log_ell)
      N <- ncol(current_log_ell)
      Q_flat <- flat$flat   # (S, N, K_total)

      Q_nat_flat <- array(0, dim = dim(Q_flat))
      for (s in seq_len(S)) {
        Q_nat_flat[s, , ] <- Q_flat[s, , ] %*% self$posterior_cov
      }
      unflatten_pytree_leaves(Q_nat_flat, flat$layout)
    },

    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_doubleprime)) {
        hess_diag <- kwargs$log_ell_doubleprime
      } else {
        hess_diag <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)
      }

      flat <- flatten_pytree_leaves(hess_diag, current_log_ell)
      hess_flat <- flat$flat   # (S, N, K_total)
      sigma_diag <- diag(self$posterior_cov)

      # div(Q) = -tr(Sigma %*% H) ~ -sum_k Sigma_kk * H_kk (diagonal-H approx)
      K_total <- dim(hess_flat)[3]
      sigma_b <- array(rep(sigma_diag, each = dim(hess_flat)[1] * dim(hess_flat)[2]),
                       dim = dim(hess_flat))
      -apply(hess_flat * sigma_b, c(1, 2), sum)
    }
  )
)


#' KL Divergence Transformation
#'
#' Q_i = -exp(log_pi - log_ell_i) * grad(log_ell_i)
#'
#' Uses the analytical divergence
#'   div(Q_KL) = c_i * (-tr(H_i) + (grad log_pi - grad log_ell_i) . grad log_ell_i)
#' where c_i = -exp(log_pi - log_ell_i) and grad log_pi ~ sum_j grad log_ell_j
#' (posterior-gradient approximation, ignoring prior). Matches the Python
#' zero-Jacobian fix.
#'
#' @export
KLDivergence <- R6::R6Class("KLDivergence",
  inherit = SmallStepTransformation,
  public = list(
    compute_Q = function(theta, data, params, current_log_ell,
                         log_pi = NULL, ...) {
      if (is.null(log_pi)) stop("log_pi required for KLDivergence")

      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }

      # log_pi: S vector; current_log_ell: S x N
      # scaling = -exp(log_pi - log_ell): S x N
      log_pi_centered <- log_pi - max(log_pi)
      scaling <- -exp(log_pi_centered - current_log_ell) # S x N

      Q <- list()
      for (k in names(grad_ll)) {
        g <- grad_ll[[k]]
        if (length(dim(g)) == 3) {
          # S x N x K: replicate scaling across K dimension
          K_dim <- dim(g)[3]
          scaling_3d <- array(rep(scaling, K_dim), dim = dim(g))
          Q[[k]] <- g * scaling_3d
        } else {
          Q[[k]] <- scaling * g
        }
      }
      return(Q)
    },

    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      log_pi <- kwargs$log_pi
      if (is.null(log_pi)) {
        return(matrix(0, nrow = nrow(current_log_ell),
                      ncol = ncol(current_log_ell)))
      }
      log_pi_centered <- log_pi - max(log_pi)
      c_i <- -exp(log_pi_centered - current_log_ell)   # (S, N)

      if (!is.null(kwargs$log_ell_doubleprime)) {
        hess_diag <- kwargs$log_ell_doubleprime
      } else {
        hess_diag <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)
      }
      tr_H <- matrix(0, nrow = nrow(current_log_ell),
                     ncol = ncol(current_log_ell))
      for (k in names(hess_diag)) {
        val <- hess_diag[[k]]
        if (length(dim(val)) == 3) {
          tr_H <- tr_H + apply(val, c(1, 2), sum)
        } else {
          tr_H <- tr_H + val
        }
      }

      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }

      # dot term: (sum_j g_j - g_i) . g_i, summed over leaves
      dot_term <- matrix(0, nrow = nrow(current_log_ell),
                         ncol = ncol(current_log_ell))
      for (k in names(grad_ll)) {
        leaf <- grad_ll[[k]]
        if (length(dim(leaf)) == 3) {
          total_g <- apply(leaf, c(1, 3), sum)              # (S, K)
          total_g_3d <- array(rep(total_g, each = 1),
                              dim = dim(leaf))
          # rep over N dimension correctly: replicate total_g across N
          total_g_3d <- aperm(array(rep(total_g,
                                        times = dim(leaf)[2]),
                                    dim = c(dim(leaf)[1], dim(leaf)[3],
                                            dim(leaf)[2])),
                              c(1, 3, 2))
          diff <- total_g_3d - leaf
          dot_term <- dot_term + apply(diff * leaf, c(1, 2), sum)
        } else {
          total_g <- rowSums(leaf)                          # (S,)
          total_g_m <- matrix(total_g, nrow = nrow(leaf),
                              ncol = ncol(leaf))
          diff <- total_g_m - leaf
          dot_term <- dot_term + diff * leaf
        }
      }

      c_i * (-tr_H + dot_term)
    }
  )
)


#' Natural-gradient KL Divergence Transformation
#'
#' Preconditions the KL gradient by the posterior covariance matrix,
#' aligning the step direction with the influence-function optimal shift.
#'
#' Q = Sigma @ Q_kl, where Sigma is the posterior covariance and
#' Q_kl is the standard KL divergence vector field.
#'
#' @export
NaturalKLDivergence <- R6::R6Class("NaturalKLDivergence",
  inherit = SmallStepTransformation,
  public = list(
    posterior_cov = NULL,

    #' @description Initialize
    #' @param likelihood_fn LikelihoodFunction object
    #' @param posterior_cov Posterior covariance matrix (K_total x K_total)
    initialize = function(likelihood_fn, posterior_cov) {
      super$initialize(likelihood_fn)
      self$posterior_cov <- posterior_cov
    },

    compute_Q = function(theta, data, params, current_log_ell,
                         log_pi = NULL, ...) {
      if (is.null(log_pi)) stop("log_pi required for NaturalKLDivergence")

      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }

      # Compute standard KL scaling: -exp(log_pi - log_ell)
      log_pi_centered <- log_pi - max(log_pi)
      scaling <- -exp(log_pi_centered - current_log_ell) # S x N

      Q_kl <- list()
      for (k in names(grad_ll)) {
        g <- grad_ll[[k]]
        if (length(dim(g)) == 3) {
          K_dim <- dim(g)[3]
          scaling_3d <- array(rep(scaling, K_dim), dim = dim(g))
          Q_kl[[k]] <- g * scaling_3d
        } else {
          Q_kl[[k]] <- scaling * g
        }
      }

      # Flatten Q_kl leaves to (S, N, K_total)
      S <- nrow(current_log_ell)
      N <- ncol(current_log_ell)

      flat_parts <- list()
      split_sizes <- integer(0)
      for (k in names(Q_kl)) {
        val <- Q_kl[[k]]
        if (length(dim(val)) == 2) {
          flat_parts[[length(flat_parts) + 1]] <- array(val, dim = c(S, N, 1))
          split_sizes <- c(split_sizes, 1L)
        } else {
          trailing <- prod(dim(val)[-(1:2)])
          flat_parts[[length(flat_parts) + 1]] <- array(val, dim = c(S, N, trailing))
          split_sizes <- c(split_sizes, as.integer(trailing))
        }
      }

      # Concatenate along last axis
      K_total <- sum(split_sizes)
      Q_flat <- array(0, dim = c(S, N, K_total))
      idx <- 1L
      for (part in flat_parts) {
        k_i <- dim(part)[3]
        Q_flat[, , idx:(idx + k_i - 1)] <- part
        idx <- idx + k_i
      }

      # Apply covariance: Q_nat[s,n,:] = Q_flat[s,n,:] %*% Sigma
      Q_nat_flat <- array(0, dim = c(S, N, K_total))
      for (s in seq_len(S)) {
        Q_nat_flat[s, , ] <- Q_flat[s, , ] %*% self$posterior_cov
      }

      # Unflatten back to list structure
      result <- list()
      idx <- 1L
      i <- 1L
      for (k in names(Q_kl)) {
        k_i <- split_sizes[i]
        orig <- Q_kl[[k]]
        if (length(dim(orig)) == 2) {
          result[[k]] <- Q_nat_flat[, , idx]
        } else {
          result[[k]] <- array(Q_nat_flat[, , idx:(idx + k_i - 1)],
                               dim = dim(orig))
        }
        idx <- idx + k_i
        i <- i + 1L
      }

      return(result)
    },

    #' @description Analytical divergence of Q_NKL = Sigma %*% Q_KL.
    #' div(Q_NKL) = c_i * (-tr(Sigma %*% H_i) + (sum_j g_j - g_i)^T Sigma g_i)
    #' with diagonal-Hessian approximation for tr(Sigma %*% H).
    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      log_pi <- kwargs$log_pi
      if (is.null(log_pi)) {
        return(matrix(0, nrow = nrow(current_log_ell),
                      ncol = ncol(current_log_ell)))
      }
      log_pi_centered <- log_pi - max(log_pi)
      c_i <- -exp(log_pi_centered - current_log_ell)

      if (!is.null(kwargs$log_ell_doubleprime)) {
        hess_diag <- kwargs$log_ell_doubleprime
      } else {
        hess_diag <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)
      }
      flat_H <- flatten_pytree_leaves(hess_diag, current_log_ell)
      hess_flat <- flat_H$flat  # (S, N, K_total)
      sigma_diag <- diag(self$posterior_cov)
      sigma_b <- array(rep(sigma_diag,
                           each = dim(hess_flat)[1] * dim(hess_flat)[2]),
                       dim = dim(hess_flat))
      tr_SH <- apply(hess_flat * sigma_b, c(1, 2), sum)  # (S, N)

      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }
      flat_G <- flatten_pytree_leaves(grad_ll, current_log_ell)
      g_flat <- flat_G$flat   # (S, N, K)
      total_g <- apply(g_flat, c(1, 3), sum)   # (S, K)
      S <- dim(g_flat)[1]; N <- dim(g_flat)[2]; K <- dim(g_flat)[3]
      total_g_3d <- aperm(array(total_g, dim = c(S, K, N)), c(1, 3, 2))
      diff <- total_g_3d - g_flat   # (S, N, K)
      # diff_Sigma[s, n, :] = diff[s, n, :] %*% Sigma
      diff_Sigma <- array(0, dim = c(S, N, K))
      for (s in seq_len(S)) {
        diff_Sigma[s, , ] <- diff[s, , ] %*% self$posterior_cov
      }
      dot_term <- apply(diff_Sigma * g_flat, c(1, 2), sum)  # (S, N)

      c_i * (-tr_SH + dot_term)
    }
  )
)


#' PMM1 (Partial Moment Matching 1) - Shift based SmallStep transformation
#'
#' Q = mean_w - mean, applied per observation as a small step.
#'
#' Analytical divergence: Q is independent of theta, so div(Q) = 0 exactly.
#'
#' @export
PMM1 <- R6::R6Class("PMM1",
  inherit = SmallStepTransformation,
  public = list(
    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      matrix(0, nrow = nrow(current_log_ell), ncol = ncol(current_log_ell))
    },

    compute_Q = function(theta, data, params, current_log_ell,
                         log_ell_original = NULL, ...) {
      if (is.null(log_ell_original)) stop("log_ell_original required for PMM1")

      log_w <- -log_ell_original
      weights <- exp(log_w)

      moments <- self$compute_moments(params, weights)

      S <- nrow(current_log_ell)
      N <- ncol(current_log_ell)

      Q <- list()
      for (name in names(params)) {
        val <- params[[name]]
        m <- moments[[name]]

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          # Scalar: S -> broadcast to S x N
          diff <- -m$mean + m$mean_w  # N
          Q[[name]] <- matrix(rep(diff, each = S), nrow = S, ncol = N)
        } else {
          # S x K -> broadcast to S x N x K
          K <- ncol(val)
          diff <- sweep(-matrix(m$mean, nrow = N, ncol = K, byrow = TRUE),
                        c(1, 2), m$mean_w, "+") # N x K
          arr <- array(0, dim = c(S, N, K))
          for (i in seq_len(N)) {
            arr[, i, ] <- matrix(diff[i, ], nrow = S, ncol = K, byrow = TRUE)
          }
          Q[[name]] <- arr
        }
      }
      return(Q)
    }
  )
)


#' PMM2 (Partial Moment Matching 2) - Scale + Shift SmallStep transformation
#'
#' Q = (ratio - 1) * val + (mean_w - ratio * mean)
#' where ratio = sqrt(var_w / var)
#'
#' @export
PMM2 <- R6::R6Class("PMM2",
  inherit = SmallStepTransformation,
  public = list(
    # Cache moments for divergence computation
    .cached_moments = NULL,

    compute_Q = function(theta, data, params, current_log_ell,
                         log_ell_original = NULL, ...) {
      if (is.null(log_ell_original)) stop("log_ell_original required for PMM2")

      log_w <- -log_ell_original
      weights <- exp(log_w)

      moments <- self$compute_moments(params, weights)
      self$.cached_moments <- moments

      S <- nrow(current_log_ell)
      N <- ncol(current_log_ell)

      Q <- list()
      for (name in names(params)) {
        val <- params[[name]]
        m <- moments[[name]]

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          ratio <- sqrt(m$var_w / (m$var + 1e-10)) # N
          term1 <- outer(val, ratio - 1, "*")
          term2 <- m$mean_w - ratio * m$mean
          Q[[name]] <- sweep(term1, 2, term2, "+")
        } else {
          K <- ncol(val)
          var_expanded <- matrix(m$var, nrow = N, ncol = K, byrow = TRUE)
          ratio <- sqrt(m$var_w / (var_expanded + 1e-10))

          arr <- array(0, dim = c(S, N, K))
          val_centered <- sweep(val, 2, m$mean, "-")
          for (i in seq_len(N)) {
            scaled <- sweep(val_centered, 2, ratio[i, ] - 1, "*")
            offset <- m$mean_w[i, ] - ratio[i, ] * m$mean
            arr[, i, ] <- sweep(scaled, 2, offset, "+")
          }
          Q[[name]] <- arr
        }
      }
      return(Q)
    },

    #' @description Compute divergence of PMM2's Q.
    #' Q_k = (ratio_k - 1) * theta_k + const, so dQ_k/dtheta_k = ratio_k - 1.
    #' div(Q) = sum_k (ratio_k - 1).
    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      moments <- self$.cached_moments
      if (is.null(moments)) {
        # Fallback: recompute
        kwargs <- list(...)
        log_ell_original <- kwargs$log_ell_original
        if (is.null(log_ell_original)) log_ell_original <- current_log_ell
        log_w <- -log_ell_original
        moments <- self$compute_moments(params, exp(log_w))
      }

      S <- nrow(current_log_ell)
      N <- ncol(current_log_ell)
      div_Q <- matrix(0, nrow = S, ncol = N)

      for (name in names(params)) {
        val <- params[[name]]
        m <- moments[[name]]

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          ratio <- sqrt(m$var_w / (m$var + 1e-10)) # N
          # Each scalar param contributes (ratio - 1) to divergence
          div_Q <- sweep(div_Q, 2, ratio - 1, "+")
        } else {
          K <- ncol(val)
          var_expanded <- matrix(m$var, nrow = N, ncol = K, byrow = TRUE)
          ratio <- sqrt(m$var_w / (var_expanded + 1e-10)) # N x K
          # Sum over K: each entry contributes (ratio_k - 1)
          div_Q <- sweep(div_Q, 2, rowSums(ratio - 1), "+")
        }
      }

      return(div_Q)
    }
  )
)


#' MM1 (Moment Matching 1) - Global shift transformation
#'
#' @export
MM1 <- R6::R6Class("MM1",
  inherit = Transformation,
  public = list(
    #' @description Apply MM1 transformation
    call = function(max_iter, params, theta, data, log_ell,
                    log_ell_original = NULL, log_pi = NULL,
                    variational = FALSE, surrogate_log_prob_fn = NULL, ...) {
      if (is.null(log_ell_original)) log_ell_original <- log_ell

      log_w <- -log_ell_original
      weights <- exp(log_w)

      moments <- self$compute_moments(params, weights)

      S <- nrow(log_ell)
      N <- ncol(log_ell)

      new_params <- list()
      for (name in names(params)) {
        val <- params[[name]]
        m <- moments[[name]]

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          diff <- -m$mean + m$mean_w  # N
          new_params[[name]] <- outer(val, rep(1, N)) + outer(rep(1, S), diff)
        } else {
          K <- ncol(val)
          diff <- sweep(-matrix(m$mean, nrow = N, ncol = K, byrow = TRUE),
                        c(1, 2), m$mean_w, "+")
          arr <- array(0, dim = c(S, N, K))
          for (i in seq_len(N)) {
            arr[, i, ] <- sweep(val, 2, diff[i, ], "+")
          }
          new_params[[name]] <- arr
        }
      }

      log_jacobian <- matrix(0, nrow = S, ncol = N)

      iw <- self$compute_importance_weights(
        self$likelihood_fn, data, params, new_params,
        log_jacobian, variational, log_pi, log_ell_original,
        surrogate_log_prob_fn
      )

      log_ell_new <- iw$log_ell_new
      exp_log_ell_new <- exp(log_ell_new)

      list(
        theta_new = new_params,
        log_jacobian = log_jacobian,
        eta_weights = iw$eta_weights,
        psis_weights = iw$psis_weights,
        khat = iw$khat,
        log_ell_new = log_ell_new,
        weight_entropy = entropy(iw$eta_weights),
        psis_entropy = entropy(iw$psis_weights),
        p_loo_eta = colSums(iw$eta_weights * exp_log_ell_new),
        p_loo_psis = colSums(iw$psis_weights * exp_log_ell_new),
        ll_loo_eta = colSums(iw$eta_weights * exp_log_ell_new),
        ll_loo_psis = colSums(iw$psis_weights * exp_log_ell_new)
      )
    }
  )
)


#' MM2 (Moment Matching 2) - Global shift and scale transformation
#'
#' @export
MM2 <- R6::R6Class("MM2",
  inherit = Transformation,
  public = list(
    #' @description Apply MM2 transformation
    call = function(max_iter, params, theta, data, log_ell,
                    log_ell_original = NULL, log_pi = NULL,
                    variational = FALSE, surrogate_log_prob_fn = NULL, ...) {
      if (is.null(log_ell_original)) log_ell_original <- log_ell

      log_w <- -log_ell_original
      weights <- exp(log_w)

      moments <- self$compute_moments(params, weights)

      S <- nrow(log_ell)
      N <- ncol(log_ell)

      new_params <- list()
      log_det_jac <- matrix(0, nrow = S, ncol = N)

      for (name in names(params)) {
        val <- params[[name]]
        m <- moments[[name]]

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          ratio <- sqrt(m$var_w / (m$var + 1e-10))
          ratio[!is.finite(ratio)] <- 1  # clamp degenerate ratios
          term1 <- outer(val - m$mean, ratio)
          new_params[[name]] <- sweep(term1, 2, m$mean_w, "+")
          log_det_jac <- sweep(log_det_jac, 2, log(ratio), "+")

        } else {
          K <- ncol(val)
          var_expanded <- matrix(m$var, nrow = N, ncol = K, byrow = TRUE)
          ratio <- sqrt(m$var_w / (var_expanded + 1e-10))
          ratio[!is.finite(ratio)] <- 1  # clamp degenerate ratios

          val_centered <- sweep(val, 2, m$mean, "-")
          arr <- array(0, dim = c(S, N, K))
          for (i in seq_len(N)) {
            scaled <- sweep(val_centered, 2, ratio[i, ], "*")
            arr[, i, ] <- sweep(scaled, 2, m$mean_w[i, ], "+")
          }
          new_params[[name]] <- arr

          log_det_k <- rowSums(log(ratio))  # N
          log_det_jac <- sweep(log_det_jac, 2, log_det_k, "+")
        }
      }

      iw <- self$compute_importance_weights(
        self$likelihood_fn, data, params, new_params,
        log_det_jac, variational, log_pi, log_ell_original,
        surrogate_log_prob_fn
      )

      log_ell_new <- iw$log_ell_new
      exp_log_ell_new <- exp(log_ell_new)

      list(
        theta_new = new_params,
        log_jacobian = log_det_jac,
        eta_weights = iw$eta_weights,
        psis_weights = iw$psis_weights,
        khat = iw$khat,
        log_ell_new = log_ell_new,
        weight_entropy = entropy(iw$eta_weights),
        psis_entropy = entropy(iw$psis_weights),
        p_loo_eta = colSums(iw$eta_weights * exp_log_ell_new),
        p_loo_psis = colSums(iw$psis_weights * exp_log_ell_new),
        ll_loo_eta = colSums(iw$eta_weights * exp_log_ell_new),
        ll_loo_psis = colSums(iw$psis_weights * exp_log_ell_new)
      )
    }
  )
)


# ============================================================================
# Full-rank transformations: MM3, PMM3
# ============================================================================

# Internal helper: flatten a named list of parameters of shape (S, ...) into
# a (S, D) matrix plus a layout descriptor for unflattening.
.flatten_params <- function(params) {
  S <- NULL
  layout <- list()
  parts <- list()
  for (name in sort(names(params))) {
    v <- params[[name]]
    if (is.null(dim(v))) {
      if (is.null(S)) S <- length(v)
      parts[[length(parts) + 1]] <- matrix(v, ncol = 1)
      layout[[length(layout) + 1]] <- list(name = name, trailing = integer(0),
                                            n_elems = 1L)
    } else {
      if (is.null(S)) S <- dim(v)[1]
      trailing <- dim(v)[-1]
      k <- as.integer(prod(trailing))
      parts[[length(parts) + 1]] <- matrix(v, nrow = dim(v)[1], ncol = k)
      layout[[length(layout) + 1]] <- list(name = name, trailing = trailing,
                                            n_elems = k)
    }
  }
  list(flat = do.call(cbind, parts), layout = layout, S = S)
}

# Unflatten a (S, N, D) array back into a named list, broadcasting where the
# original parameter was scalar.
.unflatten_to_per_obs <- function(arr_snd, layout) {
  out <- list()
  idx <- 1L
  for (entry in layout) {
    k_i <- entry$n_elems
    chunk <- arr_snd[, , idx:(idx + k_i - 1), drop = FALSE]
    if (length(entry$trailing) == 0) {
      out[[entry$name]] <- matrix(chunk, nrow = dim(arr_snd)[1],
                                  ncol = dim(arr_snd)[2])
    } else {
      out[[entry$name]] <- array(chunk,
                                 dim = c(dim(arr_snd)[1], dim(arr_snd)[2],
                                         entry$trailing))
    }
    idx <- idx + k_i
  }
  out
}


#' PMM3 (Partial Moment Matching 3) - full-rank affine with step size
#'
#' Generalization of MM3 with a tunable step size `h`. The vector field is
#' \code{Q(theta) = (L_w \%*\% Linv - I) \%*\% (theta - mu) + (mu_w - mu)},
#' so `T(theta) = theta + h * Q(theta)` interpolates between identity
#' (`h=0`) and full MM3 (`h=1`). Divergence is exact:
#' \code{div(Q) = trace(L_w \%*\% Linv - I)}, constant in `theta`.
#'
#' @export
PMM3 <- R6::R6Class("PMM3",
  inherit = SmallStepTransformation,
  public = list(
    .cached_A = NULL,

    compute_Q = function(theta, data, params, current_log_ell,
                         log_ell_original = NULL, ...) {
      if (is.null(log_ell_original))
        stop("log_ell_original required for PMM3")

      log_w <- -log_ell_original
      weights <- exp(log_w)
      S <- nrow(log_w); N <- ncol(log_w)

      fl <- .flatten_params(params)
      theta_flat <- fl$flat    # (S, D)
      D <- ncol(theta_flat)

      mu <- colMeans(theta_flat)
      centered <- sweep(theta_flat, 2, mu, "-")
      cov_u <- crossprod(centered) / S + 1e-8 * diag(D)
      L <- t(chol(cov_u))           # lower triangular
      L_inv <- solve(L)

      w_norm <- sweep(weights, 2, colSums(weights) + 1e-10, "/")
      mu_w <- t(w_norm) %*% theta_flat                # (N, D)

      # Weighted covariance per observation: cov_w[n] = Sum_s w_norm[s,n] dd^T
      cov_w <- array(0, dim = c(N, D, D))
      for (n in seq_len(N)) {
        d <- sweep(theta_flat, 2, mu_w[n, ], "-")
        cov_w[n, , ] <- t(d) %*% (d * w_norm[, n]) + 1e-8 * diag(D)
      }

      A <- array(0, dim = c(N, D, D))
      mu_shift <- sweep(mu_w, 2, mu, "-")    # (N, D)
      Q_flat <- array(0, dim = c(S, N, D))
      for (n in seq_len(N)) {
        L_w_n <- t(chol(cov_w[n, , ]))
        A_n <- L_w_n %*% L_inv - diag(D)
        A[n, , ] <- A_n
        Q_flat[, n, ] <- centered %*% t(A_n) + matrix(mu_shift[n, ],
                                                       nrow = S, ncol = D,
                                                       byrow = TRUE)
      }
      self$.cached_A <- A
      .unflatten_to_per_obs(Q_flat, fl$layout)
    },

    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      A <- self$.cached_A
      if (is.null(A)) {
        return(matrix(0, nrow = nrow(current_log_ell),
                      ncol = ncol(current_log_ell)))
      }
      N <- dim(A)[1]
      trace_A <- numeric(N)
      for (n in seq_len(N)) trace_A[n] <- sum(diag(A[n, , ]))
      matrix(trace_A, nrow = nrow(current_log_ell), ncol = N, byrow = TRUE)
    }
  )
)


#' MM3 (Moment Matching 3) - global full-rank affine transformation
#'
#' From Paananen et al. (2021). Matches the full covariance structure (not
#' just marginal variances like MM2) via
#' \code{T_i(theta) = L_w_i \%*\% Linv \%*\% (theta - mu) + mu_w_i}
#' where `L` and `L_w_i` are Cholesky factors of the unweighted and weighted
#' covariance matrices respectively, and `Linv` is the inverse of `L`. The
#' exact log-Jacobian per observation is
#' \code{log|J_i| = sum(log(diag(L_w_i))) - sum(log(diag(L)))}.
#'
#' @export
MM3 <- R6::R6Class("MM3",
  inherit = Transformation,
  public = list(
    call = function(max_iter, params, theta, data, log_ell,
                    log_ell_original = NULL, log_pi = NULL,
                    variational = FALSE, surrogate_log_prob_fn = NULL, ...) {
      if (is.null(log_ell_original)) log_ell_original <- log_ell
      log_w <- -log_ell_original
      weights <- exp(log_w)
      S <- nrow(log_w); N <- ncol(log_w)

      fl <- .flatten_params(params)
      theta_flat <- fl$flat
      D <- ncol(theta_flat)

      mu <- colMeans(theta_flat)
      centered <- sweep(theta_flat, 2, mu, "-")
      cov_u <- crossprod(centered) / S + 1e-8 * diag(D)
      L <- t(chol(cov_u))
      L_inv <- solve(L)
      log_det_L <- sum(log(diag(L)))
      z <- centered %*% t(L_inv)   # (S, D)

      w_norm <- sweep(weights, 2, colSums(weights) + 1e-10, "/")
      mu_w <- t(w_norm) %*% theta_flat     # (N, D)

      log_jac <- matrix(0, nrow = S, ncol = N)
      theta_new_flat <- array(0, dim = c(S, N, D))
      for (n in seq_len(N)) {
        d <- sweep(theta_flat, 2, mu_w[n, ], "-")
        cov_w_n <- t(d) %*% (d * w_norm[, n]) + 1e-8 * diag(D)
        L_w_n <- t(chol(cov_w_n))
        theta_new_flat[, n, ] <- z %*% t(L_w_n) + matrix(mu_w[n, ],
                                                         nrow = S, ncol = D,
                                                         byrow = TRUE)
        log_jac[, n] <- sum(log(diag(L_w_n))) - log_det_L
      }
      new_params <- .unflatten_to_per_obs(theta_new_flat, fl$layout)

      iw <- self$compute_importance_weights(
        self$likelihood_fn, data, params, new_params,
        log_jac, variational, log_pi, log_ell_original,
        surrogate_log_prob_fn
      )
      log_ell_new <- iw$log_ell_new
      exp_log_ell_new <- exp(log_ell_new)
      list(
        theta_new = new_params,
        log_jacobian = log_jac,
        eta_weights = iw$eta_weights,
        psis_weights = iw$psis_weights,
        khat = iw$khat,
        log_ell_new = log_ell_new,
        weight_entropy = entropy(iw$eta_weights),
        psis_entropy = entropy(iw$psis_weights),
        p_loo_eta = colSums(iw$eta_weights * exp_log_ell_new),
        p_loo_psis = colSums(iw$psis_weights * exp_log_ell_new),
        ll_loo_eta = colSums(iw$eta_weights * exp_log_ell_new),
        ll_loo_psis = colSums(iw$psis_weights * exp_log_ell_new)
      )
    }
  )
)


#' Variance-based Transformation
#'
#' Q = pi * (f/ell)^2 * grad(log(f/ell))
#'   = pi * exp(2 log f - 2 log ell) * (grad log f - grad log ell)
#'
#' For the default target f = ell, this collapses to grad log f - grad log ell
#' which is zero, so callers typically supply a custom f_fn (e.g. an
#' expectation target).
#'
#' Analytical divergence (with default f = ell, so delta_g = 0):
#'   div(Q_Var) ~ w * (-tr(H) - 2 * ||grad log ell||^2)
#'
#' @export
Variance <- R6::R6Class("Variance",
  inherit = SmallStepTransformation,
  public = list(
    f_fn = NULL,

    #' @description Initialize
    #' @param likelihood_fn LikelihoodFunction
    #' @param f_fn Optional function(data, params) -> (S, N) target.
    initialize = function(likelihood_fn, f_fn = NULL) {
      super$initialize(likelihood_fn)
      self$f_fn <- f_fn
    },

    compute_Q = function(theta, data, params, current_log_ell,
                         log_pi = NULL, ...) {
      if (is.null(log_pi)) stop("log_pi required for Variance transform")

      kwargs <- list(...)
      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }

      # Default target f = exp(log_ell): log_f == current_log_ell,
      # grad_log_f == grad_ll, so delta_g = 0 and Q = 0.
      if (is.null(self$f_fn)) {
        return(lapply(grad_ll, function(x) array(0, dim = dim(x))))
      }

      log_f <- log(do.call(self$f_fn, list(data, params)))   # (S, N)
      log_pi_centered <- log_pi - max(log_pi)
      log_w <- log_pi_centered + 2 * log_f - 2 * current_log_ell
      w <- exp(log_w)

      grad_log_f <- if (!is.null(kwargs$grad_log_f)) kwargs$grad_log_f else grad_ll

      Q <- list()
      for (k in names(grad_ll)) {
        gl <- grad_ll[[k]]
        gf <- grad_log_f[[k]]
        diff <- gf - gl
        if (length(dim(diff)) == 3) {
          K_dim <- dim(diff)[3]
          w_3d <- array(rep(w, K_dim), dim = dim(diff))
          Q[[k]] <- w_3d * diff
        } else {
          Q[[k]] <- w * diff
        }
      }
      Q
    },

    compute_divergence_Q = function(theta, data, params, current_log_ell, ...) {
      kwargs <- list(...)
      log_pi <- kwargs$log_pi
      if (is.null(log_pi)) {
        return(matrix(0, nrow = nrow(current_log_ell),
                      ncol = ncol(current_log_ell)))
      }
      log_pi_centered <- log_pi - max(log_pi)
      log_f <- if (is.null(self$f_fn)) current_log_ell else
        log(do.call(self$f_fn, list(data, params)))
      w <- exp(log_pi_centered + 2 * log_f - 2 * current_log_ell)

      if (!is.null(kwargs$log_ell_doubleprime)) {
        hess_diag <- kwargs$log_ell_doubleprime
      } else {
        hess_diag <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)
      }
      tr_H <- matrix(0, nrow = nrow(current_log_ell),
                     ncol = ncol(current_log_ell))
      for (k in names(hess_diag)) {
        val <- hess_diag[[k]]
        if (length(dim(val)) == 3) {
          tr_H <- tr_H + apply(val, c(1, 2), sum)
        } else {
          tr_H <- tr_H + val
        }
      }

      if (!is.null(kwargs$log_ell_prime)) {
        grad_ll <- kwargs$log_ell_prime
      } else {
        grad_ll <- self$likelihood_fn$log_likelihood_gradient(data, params)
      }
      grad_sq <- matrix(0, nrow = nrow(current_log_ell),
                        ncol = ncol(current_log_ell))
      for (k in names(grad_ll)) {
        leaf <- grad_ll[[k]]
        if (length(dim(leaf)) == 3) {
          grad_sq <- grad_sq + apply(leaf^2, c(1, 2), sum)
        } else {
          grad_sq <- grad_sq + leaf^2
        }
      }
      w * (-tr_H - 2 * grad_sq)
    }
  )
)


#' MixIS - Mixture Importance Sampling for LOO-CV (Silva & Zanella 2024)
#'
#' Resamples posterior draws using the mixture proposal
#'   q_mix(theta) ~ pi(theta|D) * sum_i 1/ell(theta|d_i)
#' and reweights with finite-variance IS weights
#'   nu_i(theta*) = (1/ell(theta*|d_i)) / sum_j (1/ell(theta*|d_j)).
#' Provides a robust LOO baseline that does not require posterior smoothing.
#'
#' @export
MixIS <- R6::R6Class("MixIS",
  inherit = Transformation,
  public = list(
    n_mix_samples = NULL,

    initialize = function(likelihood_fn, n_mix_samples = NULL) {
      super$initialize(likelihood_fn)
      self$n_mix_samples <- n_mix_samples
    },

    call = function(max_iter, params, theta, data, log_ell,
                    log_ell_original = NULL, log_pi = NULL,
                    variational = FALSE, surrogate_log_prob_fn = NULL,
                    seed = 42, ...) {
      if (is.null(log_ell_original)) log_ell_original <- log_ell
      S <- nrow(log_ell_original); N <- ncol(log_ell_original)
      n_mix <- if (is.null(self$n_mix_samples)) S else self$n_mix_samples

      # log w_mix(theta_s) = log sum_i exp(-log ell_si)
      neg_log_ell <- -log_ell_original   # (S, N)
      row_max <- apply(neg_log_ell, 1, max)
      log_w_mix <- row_max + log(rowSums(exp(neg_log_ell - row_max)))   # (S,)

      # Categorical resample of size n_mix.
      log_probs <- log_w_mix - logSumExp(log_w_mix)
      probs <- exp(log_probs - max(log_probs))
      probs <- probs / sum(probs)
      set.seed(seed)
      idx <- sample.int(S, size = n_mix, replace = TRUE, prob = probs)

      resampled_params <- lapply(params, function(v) {
        if (is.null(dim(v))) v[idx] else v[idx, , drop = FALSE]
      })

      log_ell_resampled <- self$likelihood_fn$log_likelihood(
        data, resampled_params)   # (n_mix, N)

      neg_log_ell_r <- -log_ell_resampled
      row_max_r <- apply(neg_log_ell_r, 1, max)
      log_w_mix_r <- row_max_r +
        log(rowSums(exp(neg_log_ell_r - row_max_r)))
      log_nu <- neg_log_ell_r - log_w_mix_r   # (n_mix, N)

      # PSIS smoothing
      col_max <- apply(log_nu, 2, max)
      psis_res <- psislw(log_nu - rep(col_max, each = nrow(log_nu)))
      psis_weights <- psis_res$weights
      khat <- psis_res$khat

      eta_weights <- exp(log_nu - rep(col_max, each = nrow(log_nu)))
      eta_weights <- sweep(eta_weights, 2, colSums(eta_weights), "/")

      exp_log_ell_r <- exp(log_ell_resampled)
      list(
        theta_new = resampled_params,
        log_jacobian = matrix(0, nrow = n_mix, ncol = N),
        eta_weights = eta_weights,
        psis_weights = psis_weights,
        khat = khat,
        log_ell_new = log_ell_resampled,
        weight_entropy = entropy(eta_weights),
        psis_entropy = entropy(psis_weights),
        p_loo_eta = colSums(eta_weights * exp_log_ell_r),
        p_loo_psis = colSums(psis_weights * exp_log_ell_r),
        ll_loo_eta = colSums(eta_weights * exp_log_ell_r),
        ll_loo_psis = colSums(psis_weights * exp_log_ell_r)
      )
    }
  )
)
