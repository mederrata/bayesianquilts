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

    #' @description Apply transformation step
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
      diag_hess <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)

      # Sum over parameter dimensions to get (S x N)
      divs <- list()
      for (k in names(diag_hess)) {
        val <- diag_hess[[k]]
        if (length(dim(val)) == 3) {
          divs[[k]] <- -apply(val, c(1, 2), sum)
        } else {
          divs[[k]] <- -val
        }
      }

      # Sum over all params
      total_div <- divs[[1]]
      if (length(divs) > 1) {
        for (i in 2:length(divs)) {
          total_div <- total_div + divs[[i]]
        }
      }

      return(total_div)
    }
  )
)


#' KL Divergence Transformation
#'
#' Q_i = -exp(log_pi - log_ell_i) * grad(log_ell_i)
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
    }
  )
)


#' PMM1 (Partial Moment Matching 1) - Shift based SmallStep transformation
#'
#' Q = mean_w - mean, applied per observation as a small step.
#'
#' @export
PMM1 <- R6::R6Class("PMM1",
  inherit = SmallStepTransformation,
  public = list(
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
