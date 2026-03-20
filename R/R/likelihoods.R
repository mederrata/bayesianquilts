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
      X <- as.matrix(data$X)
      y <- data$y

      beta <- params$beta # S x K
      intercept <- params$intercept # S or S x 1

      if (is.vector(beta)) beta <- matrix(beta, ncol = ncol(X))
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta) # N x S
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T) # S x N

      # ll = y * eta - log(1 + exp(eta))
      term1 <- sweep(eta, 2, y, "*")
      term2 <- log(1 + exp(eta))

      return(term1 - term2)
    },

    log_likelihood_gradient = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y

      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta) # N x S
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T) # S x N

      p <- 1 / (1 + exp(-eta))
      grad_eta <- sweep(-p, 2, y, "+") # y - p: S x N

      S <- nrow(grad_eta)
      N <- ncol(grad_eta)
      K <- ncol(X)

      grad_beta <- array(0, dim = c(S, N, K))
      for (k in seq_len(K)) {
        grad_beta[, , k] <- sweep(grad_eta, 2, X[, k], "*")
      }

      grad_intercept <- grad_eta

      return(list(beta = grad_beta, intercept = grad_intercept))
    },

    log_likelihood_hessian_diag = function(data, params) {
      X <- as.matrix(data$X)
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)

      p <- 1 / (1 + exp(-eta))
      w <- -p * (1 - p) # S x N

      S <- nrow(w)
      N <- ncol(w)
      K <- ncol(X)

      hess_beta <- array(0, dim = c(S, N, K))
      for (k in seq_len(K)) {
        hess_beta[, , k] <- sweep(w, 2, X[, k]^2, "*")
      }

      hess_intercept <- w

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
    initialize = function() {},

    log_likelihood = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T) # S x N

      # ll = y * eta - exp(eta) - lgamma(y + 1)
      term1 <- sweep(eta, 2, y, "*")
      term2 <- exp(eta)
      term3 <- matrix(lgamma(y + 1), nrow = nrow(eta), ncol = ncol(eta),
                       byrow = TRUE)

      return(term1 - term2 - term3)
    },

    log_likelihood_gradient = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)

      mu <- exp(eta)
      grad_eta <- sweep(-mu, 2, y, "+") # y - mu: S x N

      S <- nrow(grad_eta)
      N <- ncol(grad_eta)
      K <- ncol(X)

      grad_beta <- array(0, dim = c(S, N, K))
      for (k in seq_len(K)) {
        grad_beta[, , k] <- sweep(grad_eta, 2, X[, k], "*")
      }
      grad_intercept <- grad_eta

      return(list(beta = grad_beta, intercept = grad_intercept))
    },

    log_likelihood_hessian_diag = function(data, params) {
      X <- as.matrix(data$X)
      beta <- params$beta
      intercept <- params$intercept
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)

      mu <- exp(eta)
      w <- -mu # d2LL/deta2 = -mu

      S <- nrow(w)
      N <- ncol(w)
      K <- ncol(X)

      hess_beta <- array(0, dim = c(S, N, K))
      for (k in seq_len(K)) {
        hess_beta[, , k] <- sweep(w, 2, X[, k]^2, "*")
      }
      hess_intercept <- w

      return(list(beta = hess_beta, intercept = hess_intercept))
    }
  )
)


#' Linear Regression Likelihood (Gaussian)
#'
#' @export
LinearRegressionLikelihood <- R6::R6Class("LinearRegressionLikelihood",
  inherit = LikelihoodFunction,
  public = list(
    initialize = function() {},

    log_likelihood = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y
      beta <- params$beta # S x K
      intercept <- params$intercept # S or S x 1
      sigma <- params$sigma # S (residual std dev)

      if (is.vector(beta)) beta <- matrix(beta, ncol = ncol(X))
      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta) # N x S
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T) # S x N

      # ll = -0.5 * log(2*pi*sigma^2) - 0.5 * (y - eta)^2 / sigma^2
      resid <- sweep(eta, 2, y, "-") # S x N

      # sigma: S vector
      log_sigma <- log(sigma)
      half_log_2pi <- 0.5 * log(2 * pi)

      ll <- -(half_log_2pi + log_sigma) - 0.5 * (resid / sigma)^2
      return(ll)
    },

    log_likelihood_gradient = function(data, params) {
      X <- as.matrix(data$X)
      y <- data$y
      beta <- params$beta
      intercept <- params$intercept
      sigma <- params$sigma

      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T) # S x N

      resid <- sweep(eta, 2, y, "-") # S x N
      # dLL/deta = -(y - eta) / sigma^2 = resid / sigma^2
      grad_eta <- -resid / sigma^2 # S x N

      S <- nrow(grad_eta)
      N <- ncol(grad_eta)
      K <- ncol(X)

      grad_beta <- array(0, dim = c(S, N, K))
      for (k in seq_len(K)) {
        grad_beta[, , k] <- sweep(grad_eta, 2, X[, k], "*")
      }
      grad_intercept <- grad_eta

      # dLL/dsigma = -1/sigma + resid^2 / sigma^3
      grad_sigma <- -1 / sigma + rowMeans(resid^2) / sigma^3

      return(list(beta = grad_beta, intercept = grad_intercept,
                  sigma = matrix(grad_sigma, nrow = S, ncol = N)))
    },

    log_likelihood_hessian_diag = function(data, params) {
      X <- as.matrix(data$X)
      beta <- params$beta
      intercept <- params$intercept
      sigma <- params$sigma

      if (is.null(dim(intercept))) intercept <- matrix(intercept, ncol = 1)

      # d2LL/deta2 = -1/sigma^2
      S <- nrow(beta)
      N <- nrow(X)
      K <- ncol(X)

      w <- matrix(-1 / sigma^2, nrow = S, ncol = N) # S x N

      hess_beta <- array(0, dim = c(S, N, K))
      for (k in seq_len(K)) {
        hess_beta[, , k] <- sweep(w, 2, X[, k]^2, "*")
      }
      hess_intercept <- w

      # d2LL/dsigma2 = 1/sigma^2 - 3*resid^2/sigma^4
      eta_T <- tcrossprod(X, beta)
      eta_T <- sweep(eta_T, 2, as.vector(intercept), "+")
      eta <- t(eta_T)
      resid <- sweep(eta, 2, data$y, "-")
      hess_sigma <- 1 / sigma^2 - 3 * resid^2 / sigma^4

      return(list(beta = hess_beta, intercept = hess_intercept,
                  sigma = hess_sigma))
    }
  )
)
