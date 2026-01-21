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
    log_likelihood_hessian_diag = function(data, params) stop("Method log_likelihood_hessian_diag not implemented")
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
    #' @param output_prefix String prefix for output keys
    #' @param ... Additional arguments
    call = function(max_iter, params, theta, data, log_ell, ...) {
      stop("Method call not implemented")
    },
    
    #' @description Compute moments (helper)
    #' @param params List of parameters
    #' @param weights Matrix of weights (S x N)
    compute_moments = function(params, weights) {
      # Normalize weights column-wise (per N)
      # weights: S x N
      w_sum <- colSums(weights)
      w_norm <- t(t(weights) / (w_sum + 1e-10)) # S x N
      
      moments <- list()
      
      for (name in names(params)) {
        val <- params[[name]] # Expect S x K or S
        
        if (is.null(dim(val)) || length(dim(val)) == 1) {
          # Scalar param: S
          # Broadcast val to S x N? No need, just use vector
          # Mean weighted: colSums(w * val) -> N
          mean_w <- colSums(w_norm * val)
          mean_u <- mean(val)
          
          # Variance
          # val - mean_w: (S) - (N) mismatch if we do directly.
          # val (S) -> (S, N)
          val_mat <- matrix(val, nrow=length(val), ncol=ncol(weights))
          mean_mat <- matrix(mean_w, nrow=length(val), ncol=ncol(weights), byrow=TRUE)
          
          var_w <- colSums(w_norm * (val_mat - mean_mat)^2)
          var_u <- mean((val - mean_u)^2)
          
          moments[[name]] <- list(mean=mean_u, mean_w=mean_w, var=var_u, var_w=var_w)
          
        } else {
          # Matrix param: S x K
          # Weighted mean -> N x K
          # einsum('sn,sk->nk', w_norm, val)
          mean_w <- t(w_norm) %*% val # N x K
          mean_u <- colMeans(val) # K
          
          # Weighted variance
          # val: S x K
          # mean_w: N x K
          # We need variance per N, K? Or aggregated?
          # Usually variance of the parameter distribution.
          # result should be N x K.
          
          # Loop or smart expansion?
          # var_w_nk = sum_s w_sn * (v_sk - m_nk)^2
          # expand v_sk to v_snk? (Large memory)
          # (v_sk - m_nk)^2 = v_sk^2 - 2*v_sk*m_nk + m_nk^2
          # sum_s w_sn * (...)
          # = sum_s w_sn * v_sk^2  - 2 * m_nk * sum_s w_sn * v_sk + m_nk^2 * sum_s w_sn
          # = (w_norm^T %*% val^2) - 2 * m_nk * (w_norm^T %*% val) + m_nk^2 * 1
          # = (mean_sq_w) - 2 * m_nk^2 + m_nk^2 = mean_sq_w - m_nk^2
          
          mean_sq_w <- t(w_norm) %*% (val^2) # N x K
          var_w <- mean_sq_w - mean_w^2
          
          # Handle numerical noise
          var_w[var_w < 0] <- 0
          
          var_u <- apply(val, 2, var) # K
          
          moments[[name]] <- list(mean=mean_u, mean_w=mean_w, var=var_u, var_w=var_w)
        }
      }
      return(moments)
    }
  )
)
