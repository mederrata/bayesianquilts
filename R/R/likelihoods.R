#' @include ais_classes.R
NULL

#' Logistic Regression Likelihood
#'
#' @export
LogisticRegressionLikelihood <- R6::R6Class("LogisticRegressionLikelihood",
  inherit = LikelihoodFunction,
  public = list(
    initialize = function() {},
    
    log_likelihood = function(data, params) {
      # data: list(X=matrix, y=vector)
      # params: list(beta=S x K, intercept=S) or similar
      
      X <- as.matrix(data$X)
      y <- data$y
      
      beta <- params$beta # S x K
      intercept <- params$intercept # S OR S x 1
      
      if (is.vector(beta)) beta <- matrix(beta, ncol=ncol(X))
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol=1)
      
      # linear predictor eta = X %*% t(beta) + intercept
      # X: N x K
      # beta: S x K -> t(beta): K x S
      # X beta^T: N x S
      
      eta_T <- tcrossprod(X, beta) # N x S
      # Add intercept (broadcast)
      # intercept is S x 1.
      # eta_T is N x S. We want to add intercept to each row? No, intercept varies by S (column).
      # sweep over columns (2).
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      
      # eta: S x N (AIS expects S x N usually)
      eta <- t(eta_T)
      
      # log_lik = y * log(p) + (1-y) * log(1-p)
      # log(p) = -log(1 + exp(-eta))
      # log(1-p) = -eta - log(1 + exp(-eta)) = -eta + log(p)
      # Simplify:
      # ll = y * eta - log(1 + exp(eta))
      
      # Broadcast y: (N) -> (S, N)
      # y_mat <- matrix(y, nrow=nrow(eta), ncol=ncol(eta), byrow=TRUE)
      
      # Using sweep is faster/cleaner
      # y * eta -> sweep across cols
      term1 <- sweep(eta, 2, y, "*")
      term2 <- log(1 + exp(eta))
      
      return(term1 - term2)
    },
    
    log_likelihood_gradient = function(data, params) {
      # Gradient w.r.t parameters
      # dLL/deta = y - p
      # p = sigmoid(eta) = 1 / (1 + exp(-eta))
      
      X <- as.matrix(data$X)
      y <- data$y
      
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol=1)
      
      eta_T <- tcrossprod(X, beta) # N x S
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+") 
      eta <- t(eta_T) # S x N
      
      p <- 1 / (1 + exp(-eta))
      
      # grad_eta = y - p (S x N)
      grad_eta <- sweep(-p, 2, y, "+") # y - p
      
      # dLL/dbeta = (y-p) * X
      # (S x N) * (N x K) ?
      # For each s, grad_beta_s = (y - p_s) %*% X -> (1 x N) %*% (N x K) -> (1 x K)
      # Result: S x K? 
      # WAIT. log_likelihood_gradient in AIS usually expects S x N x K (gradient per observation).
      # Because AIS transformation LikelihoodDescent updates theta (S x N x K).
      
      # d(LL_n)/dbeta = (y_n - p_n) * x_n
      # We need S x N x K tensor.
      # grad_eta: S x N. 
      # X: N x K.
      # Reshape:
      # grad_eta[s, n] * X[n, k]
      # Outer product per S?
      
      S <- nrow(grad_eta)
      N <- ncol(grad_eta)
      K <- ncol(X)
      
      # This expanded array is large (S*N*K). Be careful.
      # R arrays are column-major.
      
      grad_beta <- array(0, dim=c(S, N, K))
      
      # Vectorize?
      # grad_eta (S x N) . 
      # We want for each k: grad_eta * X[, k] (broadcasted)
      for (k in 1:K) {
         # grad_eta * X[, k] (N) -> sweep each row s by X[, k]
         grad_beta[, , k] <- sweep(grad_eta, 2, X[, k], "*")
      }
      
      # dLL/dintercept = y - p
      # S x N. (Scalar param)
      grad_intercept <- grad_eta
      
      return(list(beta = grad_beta, intercept = grad_intercept))
    },
    
    log_likelihood_hessian_diag = function(data, params) {
       # Diagonal hessian w.r.t to *each* param.
       # p = sigmoid(eta)
       # d2LL/deta2 = - p(1-p)
       
       # Chain rule: d2LL/dbeta_k^2 = (d2LL/deta2) * (deta/dbeta_k)^2
       # deta/dbeta_k = x_k
       # -> - p(1-p) * x_k^2
       
      X <- as.matrix(data$X)
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol=1)
      
      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)
      
      p <- 1 / (1 + exp(-eta))
      w <- -p * (1 - p) # S x N
      
      S <- nrow(w)
      N <- ncol(w)
      K <- ncol(X)
      
      hess_beta <- array(0, dim=c(S, N, K))
      for (k in 1:K) {
         # w * X[, k]^2
         hess_beta[, , k] <- sweep(w, 2, X[, k]^2, "*")
      }
      
      hess_intercept <- w # x_intercept = 1
      
      return(list(beta = hess_beta, intercept = hess_intercept))
    }
  )
)

#' Poisson Regression Likelihood
#'
#' @export
PoissonRegressionLikelihood <- R6::R6Class("PoissonRegressionLikelihood",
  inherit = LikelihoodFunction,
  public = list(
    log_likelihood = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol=1)
      
      # eta = X beta + intercept
      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T) # S x N
      
      # mu = exp(eta)
      # ll = y * eta - exp(eta) - log(factorial(y))
      # log(factorial(y)) is constant w.r.t params, derivative 0.
      
      term1 <- sweep(eta, 2, y, "*")
      term2 <- exp(eta)
      # We ignore log(y!) for gradients but needed for LL value? Yes.
      # lgamma(y + 1)
      term3 <- matrix(lgamma(y + 1), nrow=nrow(eta), ncol=ncol(eta), byrow=TRUE)
      
      return(term1 - term2 - term3)
    },
    
    log_likelihood_gradient = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol=1)
      
      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)
      
      mu <- exp(eta)
      # dLL/deta = y - mu
      grad_eta <- sweep(-mu, 2, y, "+")
      
      S <- nrow(grad_eta)
      N <- ncol(grad_eta)
      K <- ncol(X)
      
      grad_beta <- array(0, dim=c(S, N, K))
      for (k in 1:K) {
         grad_beta[, , k] <- sweep(grad_eta, 2, X[, k], "*")
      }
      grad_intercept <- grad_eta
      
      return(list(beta = grad_beta, intercept = grad_intercept))
    },
    
    log_likelihood_hessian_diag = function(data, params) {
      # d2LL/deta2 = -mu
      X <- as.matrix(data$X)
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol=1)
      
      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)
      
      mu <- exp(eta)
      w <- -mu
      
      S <- nrow(w)
      N <- ncol(w)
      K <- ncol(X)
      
      hess_beta <- array(0, dim=c(S, N, K))
      for (k in 1:K) {
         hess_beta[, , k] <- sweep(w, 2, X[, k]^2, "*")
      }
      hess_intercept <- w
      
      return(list(beta = hess_beta, intercept = hess_intercept))
    }
  )
)
