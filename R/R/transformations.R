#' @include ais_classes.R
NULL

#' Small Step Transformation Base Class
#'
#' @export
SmallStepTransformation <- R6::R6Class("SmallStepTransformation",
  inherit = Transformation,
  public = list(
    
    #' @description Normalize vector field Q
    #' @param Q List of gradient/step arrays
    #' @param theta_std List of standard deviations
    normalize_vector_field = function(Q, theta_std=NULL) {
      Q_std <- list()
      Q_keys <- names(Q)
      
      for (k in Q_keys) {
        val <- Q[[k]]
        if (!is.null(theta_std) && !is.null(theta_std[[k]])) {
          std <- theta_std[[k]]
          if (length(dim(val)) == 3) { # S x N x K
            # divide last dim by std (K)
            Q_std[[k]] <- sweep(val, 3, std + 1e-6, "/")
          } else if (length(dim(val)) == 2) { # S x N (scalar param)
            Q_std[[k]] <- val / (std + 1e-6)
          } else {
            Q_std[[k]] <- val
          }
        } else {
          Q_std[[k]] <- val
        }
      }
      
      max_mags <- list()
      for (k in Q_keys) {
        val <- Q_std[[k]]
        if (length(dim(val)) == 3) {
          max_mags[[k]] <- apply(abs(val), c(1,2), max)
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
      
      # max over S -> N
      Q_norm_max <- apply(total_max, 2, max)
      
      list(Q_std = Q_std, Q_norm_max = Q_norm_max)
    },
    
    #' @description Apply transformation step
    call = function(max_iter, params, theta, data, log_ell, hbar=1.0, theta_std=NULL, ...) {
      
      # 1. Compute Q
      Q <- self$compute_Q(theta, data, params, log_ell, ...)
      
      # 2. Normalize
      norm_res <- self$normalize_vector_field(Q, theta_std)
      Q_norm_max <- norm_res$Q_norm_max
      
      # h = hbar / norm
      h <- hbar / (Q_norm_max + 1e-8) # Vector N
      
      # 3. Update theta
      theta_new <- list()
      for (k in names(theta)) {
        t_val <- theta[[k]]
        q_val <- Q[[k]]
        
        if (length(dim(t_val)) == 3) {
          # S x N x K
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
      
      if (length(dim(div_Q)) == 2) {
         # S x N
         term <- 1.0 + sweep(div_Q, 2, h, "*")
         log_jacobian <- log(abs(term))
      } else {
        # default 0 if div_Q not computed
        log_jacobian <- matrix(0, nrow=nrow(log_ell), ncol=ncol(log_ell))
      }
      
      # Return results
      list(theta_new = theta_new, log_jacobian = log_jacobian)
    }
  )
)

#' Likelihood Descent Transformation
#'
#' @export
LikelihoodDescent <- R6::R6Class("LikelihoodDescent",
  inherit = SmallStepTransformation,
  public = list(
    compute_Q = function(theta, data, params, current_log_ell, ...) {
      # For Likelihood Descent, Q is the gradient of log likelihood
      # We assume the user implements gradient calculation that returns S x N x K structure
      # matching theta (which is S x N x K)

      # Note: params passed here might normally be S x K, but for Q computation we might need
      # effectively to evaluate gradient at 'theta' (which splits by N).
      # But basic Likelihood Descent uses the gradient at the *original* params usually?
      # No, it's iterative or uses local gradient?
      # In Python code: return self.likelihood_fn.log_likelihood_gradient(data, params)
      # It uses 'params' (the dict).
      # If 'params' is the original S x K samples, then gradient is S x N x K.
      # This matches theta structure (S x N x K).

      # However, if we were iterative (max_iter > 1), we would need gradient at theta values.
      # The python code for 'LikelihoodDescent' ignores 'theta' arg and uses 'params'.
      # This implies it takes a ONE STEP transformation from the posterior.
      # It does NOT update iteratively in the Python implementation shown (it returns grad_ll directly).

      grad <- self$likelihood_fn$log_likelihood_gradient(data, params)
      return(grad)
    },

    compute_divergence_Q = function(theta, data, params, ...) {
      # div(Q) = Laplacian(log_ell) = sum(diag(Hessian))
      diag_hess <- self$likelihood_fn$log_likelihood_hessian_diag(data, params)

      # diag_hess is list of S x N x K (or S x N)
      # Sum over K (dims > 2)
      divs <- list()
      for (k in names(diag_hess)) {
        val <- diag_hess[[k]]
        if (length(dim(val)) == 3) {
          divs[[k]] <- apply(val, c(1,2), sum)
        } else {
          divs[[k]] <- val
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


#' MM1 (Moment Matching 1) Transformation - Shift Only
#'
#' @export
MM1 <- R6::R6Class("MM1",
  inherit = Transformation,
  public = list(
    #' @description Apply MM1 transformation
    call = function(max_iter, params, theta, data, log_ell, log_ell_original = NULL, ...) {
      if (is.null(log_ell_original)) {
        log_ell_original <- log_ell
      }

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
          # Scalar param: S
          diff <- -m$mean + m$mean_w  # N
          # Broadcast val (S) + diff (N) -> S x N
          new_params[[name]] <- outer(val, rep(1, N)) + outer(rep(1, S), diff)
        } else {
          # Matrix param: S x K
          # diff: -mean (K) + mean_w (N x K) -> N x K
          K <- ncol(val)
          diff <- sweep(-matrix(m$mean, nrow=N, ncol=K, byrow=TRUE), c(1,2), m$mean_w, "+")
          # val (S x K) + diff (N x K) -> S x N x K
          arr <- array(0, dim=c(S, N, K))
          for (i in 1:N) {
            arr[, i, ] <- sweep(val, 2, diff[i, ], "+")
          }
          new_params[[name]] <- arr
        }
      }

      # MM1 Jacobian is 1 (log = 0)
      log_jacobian <- matrix(0, nrow=S, ncol=N)

      list(theta_new = new_params, log_jacobian = log_jacobian)
    }
  )
)


#' MM2 (Moment Matching 2) Transformation - Shift and Scale
#'
#' @export
MM2 <- R6::R6Class("MM2",
  inherit = Transformation,
  public = list(
    #' @description Apply MM2 transformation
    call = function(max_iter, params, theta, data, log_ell, log_ell_original = NULL, ...) {
      if (is.null(log_ell_original)) {
        log_ell_original <- log_ell
      }

      log_w <- -log_ell_original
      weights <- exp(log_w)

      moments <- self$compute_moments(params, weights)

      S <- nrow(log_ell)
      N <- ncol(log_ell)

      new_params <- list()
      log_det_jac <- matrix(0, nrow=S, ncol=N)

      for (name in names(params)) {
        val <- params[[name]]
        m <- moments[[name]]

        if (is.null(dim(val)) || length(dim(val)) == 1) {
          # Scalar param: S
          # ratio: sqrt(var_w / var) -> N
          ratio <- sqrt(m$var_w / (m$var + 1e-10))

          # term1: ratio * (val - mean) -> S x N
          term1 <- outer(val - m$mean, ratio)
          # new_val = term1 + mean_w
          new_params[[name]] <- sweep(term1, 2, m$mean_w, "+")

          # log det: log(ratio) for each N, same across S
          log_det_jac <- sweep(log_det_jac, 2, log(ratio), "+")

        } else {
          # Matrix param: S x K
          K <- ncol(val)

          # ratio: sqrt(var_w / var) -> N x K
          var_expanded <- matrix(m$var, nrow=N, ncol=K, byrow=TRUE)
          ratio <- sqrt(m$var_w / (var_expanded + 1e-10))

          # val_centered: S x K
          val_centered <- sweep(val, 2, m$mean, "-")

          # new_val[s, n, k] = ratio[n, k] * val_centered[s, k] + mean_w[n, k]
          arr <- array(0, dim=c(S, N, K))
          for (i in 1:N) {
            scaled <- sweep(val_centered, 2, ratio[i, ], "*")
            arr[, i, ] <- sweep(scaled, 2, m$mean_w[i, ], "+")
          }
          new_params[[name]] <- arr

          # log det: sum_k log(ratio[n, k])
          log_det_k <- rowSums(log(ratio))  # N
          log_det_jac <- sweep(log_det_jac, 2, log_det_k, "+")
        }
      }

      list(theta_new = new_params, log_jacobian = log_det_jac)
    }
  )
)
