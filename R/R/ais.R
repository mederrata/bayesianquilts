#' @include ais_classes.R transformations.R psis.R utils.R
NULL

#' Default transformation order by computational complexity (least to most expensive)
#' @export
DEFAULT_TRANSFORMATION_ORDER <- c(
  "identity",   # No computation - just standard PSIS-LOO
  "mm1",        # Global moment matching (shift only)
  "mm2",        # Global moment matching (shift + scale)
  "pmm1",       # Partial moment matching (shift)
  "pmm2",       # Partial moment matching (shift + scale)
  "ll",         # Likelihood descent (requires gradient)
  "kl",         # KL divergence (requires gradient + posterior weights)
  "var"         # Variance-based (requires gradient + Hessian + f_fn)
)

#' Adaptive Importance Sampler
#'
#' @export
AdaptiveImportanceSampler <- R6::R6Class("AdaptiveImportanceSampler",
  public = list(
    likelihood_fn = NULL,
    prior_log_prob_fn = NULL,

    #' @description Initialize sampler
    #' @param likelihood_fn LikelihoodFunction object
    #' @param prior_log_prob_fn Optional prior function
    initialize = function(likelihood_fn, prior_log_prob_fn = NULL) {
      self$likelihood_fn <- likelihood_fn
      self$prior_log_prob_fn <- prior_log_prob_fn
    },

    #' @description Run AIS for LOO-CV
    #'
    #' By default, transformations are tried in order of computational complexity

    #' (identity < mm1/mm2 < pmm1/pmm2 < ll < kl < var). Once a data point achieves
    #' khat < khat_threshold, it is considered "adapted" and subsequent transformations
    #' are skipped for that point (unless try_all_transformations=TRUE).
    #'
    #' @param data Data object
    #' @param params List of parameters (S x K)
    #' @param rhos Vector of step sizes
    #' @param transformations Vector of transformation names to run, in order.
    #'        If NULL, uses DEFAULT_TRANSFORMATION_ORDER (filtering unavailable ones).
    #' @param try_all_transformations If FALSE (default), skip transformations for data
    #'        points that have already achieved khat < khat_threshold.
    #'        If TRUE, try all transformations for all points.
    #' @param khat_threshold Threshold for considering a point "adapted" (default 0.7).
    #'        Points with khat below this threshold won't be processed by subsequent
    #'        transformations unless try_all_transformations=TRUE.
    #' @param verbose Boolean
    #' @return List of results
    adaptive_is_loo = function(data, params, rhos=NULL, transformations=NULL,
                               try_all_transformations=FALSE, khat_threshold=0.7,
                               verbose=FALSE) {
      
      # 1. Initial computations
      log_ell <- self$likelihood_fn$log_likelihood(data, params) # S x N
      S <- nrow(log_ell)
      N <- ncol(log_ell)

      # theta_std
      theta_std <- list()
      for (k in names(params)) {
        val <- params[[k]]
        if (is.null(dim(val))) {
          theta_std[[k]] <- sd(val)
        } else {
          theta_std[[k]] <- apply(val, 2, sd)
        }
      }

      # Expand params to theta (S x N x K)
      theta <- list()
      for (k in names(params)) {
        val <- params[[k]]
        if (is.null(dim(val))) {
           # S -> S x N
           theta[[k]] <- matrix(val, nrow=S, ncol=N)
        } else {
           # S x K -> S x N x K
           K <- ncol(val)
           # Create array
           arr <- array(0, dim=c(S, N, K))
           for (i in 1:N) {
             arr[, i, ] <- val
           }
           theta[[k]] <- arr
        }
      }

      if (is.null(rhos)) {
        rhos <- exp(seq(log(0.01), log(10), length.out=7))
      }

      # Available transformations (those actually implemented)
      available_transforms <- c("identity", "ll", "mm1", "mm2")

      # Determine transformation order
      if (is.null(transformations)) {
        # Default order by computational complexity, filtered to available
        transform_order <- intersect(DEFAULT_TRANSFORMATION_ORDER, available_transforms)
      } else {
        # User-specified order, filtered to available
        transform_order <- intersect(transformations, available_transforms)
      }

      results <- list()
      best_khat <- rep(Inf, N)
      best_res <- list(
        khat = best_khat,
        ll_loo = rep(-Inf, N),
        p_loo = rep(0, N)
      )

      # Track which points are considered "adapted" (khat < threshold)
      adapted_mask <- rep(FALSE, N)

      # Helper to update best (only for non-adapted points if not try_all)
      update_best <- function(res) {
        if (try_all_transformations) {
          improvement_mask <- res$khat < best_khat
        } else {
          improvement_mask <- (res$khat < best_khat) & (!adapted_mask)
        }

        best_khat[improvement_mask] <<- res$khat[improvement_mask]

        # Merge fields
        for (f in names(res)) {
           if (is.null(best_res[[f]])) {
             best_res[[f]] <<- res[[f]]
           } else {
             if (is.matrix(res[[f]]) || is.vector(res[[f]])) {
               # Handle dimensions
               if (length(res[[f]]) == N) {
                 best_res[[f]][improvement_mask] <<- res[[f]][improvement_mask]
               } else if (is.matrix(res[[f]]) && ncol(res[[f]]) == N) {
                 best_res[[f]][, improvement_mask] <<- res[[f]][, improvement_mask]
               }
             }
           }
        }

        # Update adapted mask
        if (!try_all_transformations) {
          newly_adapted <- best_khat < khat_threshold
          adapted_mask <<- adapted_mask | newly_adapted
        }
      }

      # Helper function to compute log_ell_new from transformed params
      compute_log_ell_new <- function(theta_new) {
        log_ell_new <- matrix(0, nrow=S, ncol=N)
        for (i in 1:N) {
          # Extract params for i
          p_i <- list()
          for (k in names(theta_new)) {
            val <- theta_new[[k]]
            if (length(dim(val)) == 3) {
              p_i[[k]] <- val[, i, ] # S x K
            } else {
              p_i[[k]] <- val[, i] # S
            }
          }
          # Evaluate likelihood (O(N^2) total but correct)
          ll_full <- self$likelihood_fn$log_likelihood(data, p_i)
          log_ell_new[, i] <- ll_full[, i]
        }
        return(log_ell_new)
      }

      # Run transformations in order of computational complexity
      for (trans_name in transform_order) {
        # Early exit if all points are adapted and we're not forcing all transformations
        if (!try_all_transformations && all(adapted_mask)) {
          if (verbose) message(sprintf("All %d points adapted, skipping remaining transformations", N))
          break
        }

        # Verbose: show remaining points
        if (verbose) {
          if (!try_all_transformations) {
            n_remaining <- N - sum(adapted_mask)
            message(sprintf("Running %s... (%d points remaining)", trans_name, n_remaining))
          } else {
            message(sprintf("Running %s...", trans_name))
          }
        }

        if (trans_name == "identity") {
          log_eta <- -log_ell

          # PSIS
          psis_res <- psislw(log_eta)

          # Metrics
          eta_weights <- exp(log_eta - logSumExp(log_eta))
          psis_weights <- psis_res$weights

          ll_loo <- colSums(psis_weights * exp(log_ell))

          res_id <- list(
             khat = psis_res$khat,
             log_weights = psis_res$log_weights,
             ll_loo = ll_loo
          )
          results[["identity"]] <- res_id
          update_best(res_id)

          if (verbose && !try_all_transformations) {
            n_adapted <- sum(adapted_mask)
            message(sprintf("  %d/%d points adapted (khat < %.2f)", n_adapted, N, khat_threshold))
          }

        } else if (trans_name == "mm1") {
          # MM1 transformation (no rho sweep - global transform)
          tryCatch({
            transform <- MM1$new(self$likelihood_fn)
            step_res <- transform$call(
              max_iter=1,
              params=params,
              theta=theta,
              data=data,
              log_ell=log_ell,
              log_ell_original=log_ell
            )

            theta_new <- step_res$theta_new
            log_jac <- step_res$log_jacobian

            log_ell_new <- compute_log_ell_new(theta_new)

            log_eta <- log_ell - log_ell_new + log_jac
            psis_res <- psislw(log_eta)

            res_mm1 <- list(
               khat = psis_res$khat,
               log_weights = psis_res$log_weights,
               ll_loo = colSums(exp(psis_res$log_weights) * exp(log_ell_new))
            )

            results[["mm1"]] <- res_mm1
            update_best(res_mm1)

          }, error = function(e) {
            if (verbose) message("Error in mm1: ", e$message)
          })

        } else if (trans_name == "mm2") {
          # MM2 transformation (no rho sweep - global transform)
          tryCatch({
            transform <- MM2$new(self$likelihood_fn)
            step_res <- transform$call(
              max_iter=1,
              params=params,
              theta=theta,
              data=data,
              log_ell=log_ell,
              log_ell_original=log_ell
            )

            theta_new <- step_res$theta_new
            log_jac <- step_res$log_jacobian

            log_ell_new <- compute_log_ell_new(theta_new)

            log_eta <- log_ell - log_ell_new + log_jac
            psis_res <- psislw(log_eta)

            res_mm2 <- list(
               khat = psis_res$khat,
               log_weights = psis_res$log_weights,
               ll_loo = colSums(exp(psis_res$log_weights) * exp(log_ell_new))
            )

            results[["mm2"]] <- res_mm2
            update_best(res_mm2)

          }, error = function(e) {
            if (verbose) message("Error in mm2: ", e$message)
          })

        } else if (trans_name == "ll") {
          transform <- LikelihoodDescent$new(self$likelihood_fn)

          for (rho in rhos) {
            # Call transformation
            tryCatch({
              step_res <- transform$call(
                 max_iter=1,
                 params=params,
                 theta=theta,
                 data=data,
                 log_ell=log_ell,
                 hbar=rho,
                 theta_std=theta_std
              )

              theta_new <- step_res$theta_new
              log_jac <- step_res$log_jacobian

              log_ell_new <- compute_log_ell_new(theta_new)

              # Weights
              log_eta <- log_ell - log_ell_new + log_jac

              psis_res <- psislw(log_eta)

              res_ll <- list(
                 khat = psis_res$khat,
                 log_weights = psis_res$log_weights,
                 ll_loo = colSums(exp(psis_res$log_weights) * exp(log_ell_new))
              )

              key <- paste0("ll_rho", sprintf("%.2e", rho))
              results[[key]] <- res_ll
              update_best(res_ll)

            }, error = function(e) {
              if (verbose) message(sprintf("Error in ll rho=%.2e: %s", rho, e$message))
            })
          }
        }
      }

      results$best <- best_res
      results
    }
  )
)
