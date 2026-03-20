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
  "nkl"         # Natural-gradient KL (requires gradient + posterior covariance)
)

#' Adaptive Importance Sampler
#'
#' @description
#' Generalized Adaptive Importance Sampling for Leave-One-Out Cross-Validation.
#'
#' Implements adaptive importance sampling transformations that work with any
#' likelihood function. Provides several transformation strategies for generating
#' importance weights, tried in order of computational complexity.
#'
#' @export
AdaptiveImportanceSampler <- R6::R6Class("AdaptiveImportanceSampler",
  public = list(
    likelihood_fn = NULL,
    prior_log_prob_fn = NULL,
    surrogate_log_prob_fn = NULL,

    #' @description Initialize sampler
    #' @param likelihood_fn LikelihoodFunction object
    #' @param prior_log_prob_fn Optional prior function
    #' @param surrogate_log_prob_fn Optional surrogate density function (for variational)
    initialize = function(likelihood_fn, prior_log_prob_fn = NULL,
                          surrogate_log_prob_fn = NULL) {
      self$likelihood_fn <- likelihood_fn
      self$prior_log_prob_fn <- prior_log_prob_fn
      self$surrogate_log_prob_fn <- surrogate_log_prob_fn
    },

    #' @description Run AIS for LOO-CV
    #'
    #' By default, transformations are tried in order of computational complexity
    #' (identity < mm1/mm2 < pmm1/pmm2 < ll < kl). Once a data point achieves
    #' khat < khat_threshold, it is considered "adapted" and subsequent
    #' transformations are skipped for that point (unless
    #' try_all_transformations=TRUE).
    #'
    #' @param data Data object
    #' @param params List of parameters (S x K)
    #' @param rhos Vector of step sizes for gradient-based transforms
    #' @param transformations Character vector of transformation names to run.
    #'        If NULL, uses DEFAULT_TRANSFORMATION_ORDER.
    #' @param try_all_transformations If FALSE (default), skip transformations
    #'        for data points that already have khat < khat_threshold.
    #' @param khat_threshold Threshold for considering a point "adapted"
    #'        (default 0.7).
    #' @param variational Whether using variational approximation
    #' @param verbose Boolean
    #' @return List with best results including khat, ll_loo_psis, p_loo_psis, etc.
    adaptive_is_loo = function(data, params, rhos = NULL,
                               transformations = NULL,
                               try_all_transformations = FALSE,
                               khat_threshold = 0.7,
                               variational = FALSE,
                               verbose = FALSE) {

      # 1. Initial computations
      log_ell <- self$likelihood_fn$log_likelihood(data, params) # S x N
      S <- nrow(log_ell)
      N <- ncol(log_ell)

      # Standard deviation of parameters (for standardization)
      theta_std <- list()
      for (k in names(params)) {
        val <- params[[k]]
        if (is.null(dim(val))) {
          s <- sd(val)
          theta_std[[k]] <- if (s < 1e-6) 1.0 else s
        } else {
          s <- apply(val, 2, sd)
          s[s < 1e-6] <- 1.0
          theta_std[[k]] <- s
        }
      }

      # Expand params to theta (S x N x K) for per-observation transforms
      theta <- list()
      for (k in names(params)) {
        val <- params[[k]]
        if (is.null(dim(val))) {
          theta[[k]] <- matrix(val, nrow = S, ncol = N)
        } else {
          K <- ncol(val)
          arr <- array(0, dim = c(S, N, K))
          for (i in seq_len(N)) {
            arr[, i, ] <- val
          }
          theta[[k]] <- arr
        }
      }

      if (is.null(rhos)) {
        rhos <- exp(seq(log(0.01), log(10), length.out = 7))
      }

      # Determine log_pi
      if (variational && !is.null(self$surrogate_log_prob_fn)) {
        log_pi <- self$surrogate_log_prob_fn(params)
      } else if (!is.null(self$prior_log_prob_fn)) {
        log_prior <- self$prior_log_prob_fn(params)
        log_pi <- rowSums(log_ell) + log_prior
      } else {
        log_pi <- rowSums(log_ell)
      }

      # Compute posterior covariance for NKL transform
      flat_parts <- list()
      for (k in names(params)) {
        val <- params[[k]]
        if (is.null(dim(val))) {
          flat_parts[[length(flat_parts) + 1]] <- matrix(val, ncol = 1)
        } else {
          flat_parts[[length(flat_parts) + 1]] <- matrix(val, nrow = nrow(val))
        }
      }
      theta_flat <- do.call(cbind, flat_parts) # S x K_total
      posterior_cov <- cov(theta_flat) + 1e-6 * diag(ncol(theta_flat))

      # Available transformations
      all_transforms <- list(
        ll = LikelihoodDescent$new(self$likelihood_fn),
        kl = KLDivergence$new(self$likelihood_fn),
        nkl = NaturalKLDivergence$new(self$likelihood_fn, posterior_cov),
        pmm1 = PMM1$new(self$likelihood_fn),
        pmm2 = PMM2$new(self$likelihood_fn),
        mm1 = MM1$new(self$likelihood_fn),
        mm2 = MM2$new(self$likelihood_fn)
      )

      # Determine transformation order
      if (is.null(transformations)) {
        transform_order <- DEFAULT_TRANSFORMATION_ORDER
      } else {
        transform_order <- transformations
      }
      # Filter to available
      transform_order <- transform_order[
        transform_order %in% c(names(all_transforms), "identity")
      ]

      results <- list()
      best_khat <- rep(Inf, N)
      best_res <- list(
        khat = best_khat,
        ll_loo_eta = rep(-Inf, N),
        ll_loo_psis = rep(-Inf, N),
        p_loo_eta = rep(0, N),
        p_loo_psis = rep(0, N)
      )

      adapted_mask <- rep(FALSE, N)

      # Helper to update best results
      update_best <- function(res) {
        if (try_all_transformations) {
          improvement <- res$khat < best_khat
        } else {
          improvement <- (res$khat < best_khat) & (!adapted_mask)
        }

        best_khat[improvement] <<- res$khat[improvement]
        for (f in c("ll_loo_eta", "ll_loo_psis", "p_loo_eta", "p_loo_psis")) {
          if (!is.null(res[[f]]) && length(res[[f]]) == N) {
            best_res[[f]][improvement] <<- res[[f]][improvement]
          }
        }
        best_res$khat <<- best_khat

        if (!try_all_transformations) {
          adapted_mask <<- adapted_mask | (best_khat < khat_threshold)
        }
      }

      # Run transformations
      for (trans_name in transform_order) {
        # Early exit
        if (!try_all_transformations && all(adapted_mask)) {
          if (verbose) message(sprintf("All %d points adapted, stopping", N))
          break
        }

        if (verbose) {
          n_remaining <- if (!try_all_transformations) N - sum(adapted_mask) else N
          message(sprintf("Running %s... (%d points remaining)", trans_name,
                          n_remaining))
        }

        if (trans_name == "identity") {
          log_eta <- -log_ell
          psis_res <- psislw(log_eta)

          eta_weights <- exp(log_eta - t(replicate(S, logSumExp(log_eta))))
          exp_ll <- exp(log_ell)

          res_id <- list(
            khat = psis_res$khat,
            ll_loo_eta = colSums(eta_weights * exp_ll),
            ll_loo_psis = colSums(psis_res$weights * exp_ll),
            p_loo_eta = colSums(eta_weights * exp_ll),
            p_loo_psis = colSums(psis_res$weights * exp_ll)
          )
          results[["identity"]] <- res_id
          update_best(res_id)

          if (verbose) {
            message(sprintf("  %d/%d adapted (khat < %.2f)",
                            sum(adapted_mask), N, khat_threshold))
          }

        } else if (trans_name %in% c("mm1", "mm2")) {
          # Global transforms (no rho sweep)
          tryCatch({
            transform <- all_transforms[[trans_name]]
            res <- transform$call(
              max_iter = 1, params = params, theta = theta,
              data = data, log_ell = log_ell,
              log_ell_original = log_ell, log_pi = log_pi,
              variational = variational,
              surrogate_log_prob_fn = self$surrogate_log_prob_fn
            )
            results[[trans_name]] <- res
            update_best(res)
          }, error = function(e) {
            if (verbose) message(sprintf("Error in %s: %s", trans_name, e$message))
          })

        } else if (trans_name %in% c("ll", "kl", "nkl", "pmm1", "pmm2")) {
          # Small-step transforms with rho sweep
          transform <- all_transforms[[trans_name]]

          for (rho in rhos) {
            tryCatch({
              res <- transform$call(
                max_iter = 1, params = params, theta = theta,
                data = data, log_ell = log_ell,
                hbar = rho, theta_std = theta_std,
                log_ell_original = log_ell, log_pi = log_pi,
                variational = variational,
                surrogate_log_prob_fn = self$surrogate_log_prob_fn
              )
              key <- sprintf("%s_rho%.2e", trans_name, rho)
              results[[key]] <- res
              update_best(res)
            }, error = function(e) {
              if (verbose) {
                message(sprintf("Error in %s rho=%.2e: %s",
                                trans_name, rho, e$message))
              }
            })
          }
        }
      }

      results$best <- best_res
      results
    }
  )
)
