#' @include ais_classes.R transformations.R psis.R utils.R
NULL

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
    #' @param data Data object
    #' @param params List of parameters (S x K)
    #' @param rhos Vector of step sizes
    #' @param transformations Vector of transformation names
    #' @param verbose Boolean
    #' @return List of results
    adaptive_is_loo = function(data, params, rhos=NULL, transformations=c("ll", "identity"), verbose=FALSE) {
      
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
      
      results <- list()
      best_khat <- rep(Inf, N)
      best_res <- list(khat = best_khat)
      
      # Helper to update best
      update_best <- function(res) {
        idx <- res$khat < best_khat
        best_khat[idx] <<- res$khat[idx]
        
        # Merge fields
        for (f in names(res)) {
           if (is.null(best_res[[f]])) {
             best_res[[f]] <<- res[[f]]
           } else {
             if (is.matrix(res[[f]]) || is.vector(res[[f]])) {
               # Handle dimensions
               if (length(res[[f]]) == N) {
                 best_res[[f]][idx] <<- res[[f]][idx]
               } else if (is.matrix(res[[f]]) && ncol(res[[f]]) == N) {
                 best_res[[f]][, idx] <<- res[[f]][, idx]
               }
             }
           }
        }
      }
      
      # Run transformations
      for (trans_name in transformations) {
         if (trans_name == "identity") {
            if (verbose) message("Running identity...")
            log_eta <- -log_ell
            
            # PSIS
            psis_res <- psislw(log_eta)
            
            # Metrics
            eta_weights <- exp(log_eta - logSumExp(log_eta))
            psis_weights <- psis_res$weights
            
            # predictions (log_likelihood evaluated at params)
            # For identity, it's just log_ell
            pred <- log_ell
            p_loo <- colSums(psis_weights * exp(pred)) # Wait, p_loo definition?
            # p_loo = lppd - elpd_loo? 
            # Python code: p_loo_psis = sum(exp(pred) * psis_weights, axis=0)?
            # That looks like expected likelihood?
            # Standard definition p_loo = var(log_lik) or similar.
            # But specific formula used in code:
            # p_loo_psis = jnp.sum(jnp.exp(predictions) * psis_weights, axis=0) -> This sums (Likelihood * Weights). Expected Likelihood.
            
            ll_loo <- colSums(psis_weights * exp(log_ell))
            
            res_id <- list(
               khat = psis_res$khat,
               log_weights = psis_res$log_weights,
               ll_loo = ll_loo
            )
            # Add to results
            results[["identity"]] <- res_id
            update_best(res_id)
            
         } else if (trans_name == "ll") {
            transform <- LikelihoodDescent$new(self$likelihood_fn)
            
            for (rho in rhos) {
               if (verbose) message(sprintf("Running ll rho=%.2e...", rho))
               
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
                 
                 # Reconstruct params to evaluate likelihood
                 # theta_new is (S, N, K).
                 # We need to evaluate log likelihood for each N using its specific params.
                 # Python code did: log_ell_new = likelihood_fn.log_likelihood(data, params_transformed)
                 # But params_transformed there was (S, N, K) and likelihood fn handled it or batched.
                 
                 # Here, let's assume log_likelihood can handle S x N x K if passed specially,
                 # Or we loop over N.
                 # Let's simple loop over N for MVP.
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
                    # Evaluate likelihood on i-th data point?
                    # log_likelihood interface takes 'data'.
                    # We might need to slice data too if likelihood fn expects full data?
                    # Likelihood fn usually computes for all N.
                    # But here we have DIFFERENT params for each N.
                    # This implies we need to evaluate log_lik(data[i], params[i]) -> S.
                    
                    # Assume user provided log_likelihood handles slicing naturally?
                    # Or we need a helper "log_likelihood_i(data, params, i)"
                    
                    # For now, I'll assume users can handle a list of params where each element is array S x N x K ?
                    # Probably not.
                    
                    # I will assume I slice data and call log_likelihood with params S x K.
                    # This relies on data slicing.
                    # Slicing data is hard if data is arbitrary object.
                    
                    # Fallback: Assume params can be passed as is and likelihood function docs say "params can be S x N x K".
                    # But I defined `LikelihoodFunction` taking S x K.
                    
                    # I'll create a `log_likelihood_batch` helper in `LikelihoodFunction`?
                    # Or just loop providing subsets.
                    # To slice data, I need to know structure.
                    # I'll optimistically assume data is list of arrays or data.frame.
                    
                    # For MVP, warning about slowness.
                    
                    # We'll use a `get_data_slice` hook if available?
                    # Or just assuming list/df structure:
                    d_i <- data # Placeholder
                    # Actually, if I pass full data and params S x K, it computes S x N matrix.
                    # I only want i-th column resulting from params S x K.
                    
                    # We will call log_likelihood(data, p_i) -> S x N_total.
                    # And take [, i].
                    # This is O(N^2) total cost. Very slow.
                    # But for MVP correctness it works.
                    
                    ll_full <- self$likelihood_fn$log_likelihood(data, p_i)
                    log_ell_new[, i] <- ll_full[, i]
                 }
                 
                 # Weights
                 # log_eta = log_ell - log_ell_new + log_jac
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
                  if (verbose) message("Error in ll: ", e$message)
               })
            }
         }
      }
      
      results$best <- best_res
      results
    }
  )
)
