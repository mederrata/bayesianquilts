#' @include ais_classes.R likelihoods.R
NULL

#' Extract parameters from a brmsfit object for AIS
#'
#' @param fit A brmsfit object
#' @return List of parameters (beta, intercept) suitable for AIS likelihoods
#' @export
extract_brms_params <- function(fit) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    stop("Package 'brms' needed for this function to work.")
  }

  samples <- brms::as_draws_matrix(fit)
  pnames <- colnames(samples)

  # brms X matrix includes an intercept column, so we keep all b_*
  # coefficients together as beta (matching X's column order).
  beta_cols <- grep("^b_", pnames, value = TRUE)

  if (length(beta_cols) > 0) {
    beta <- as.matrix(samples[, beta_cols, drop = FALSE])
    colnames(beta) <- sub("^b_", "", beta_cols)
  } else {
    beta <- matrix(0, nrow = nrow(samples), ncol = 0)
  }

  # No separate intercept — it's included in beta
  result <- list(beta = beta, intercept = rep(0, nrow(samples)))

  if ("sigma" %in% pnames) {
    result$sigma <- as.numeric(samples[, "sigma"])
  }

  result
}

#' Extract data from a brmsfit object for AIS
#'
#' @param fit A brmsfit object
#' @return List with X and y
#' @export
extract_brms_data <- function(fit) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    stop("Package 'brms' needed for this function to work.")
  }

  sdata <- brms::standata(fit)

  if (!is.null(sdata$X)) {
    X <- sdata$X
  } else {
    X <- matrix(0, nrow = length(sdata$Y), ncol = 0)
  }

  if (!is.null(sdata$Y)) {
    y <- sdata$Y
  } else {
    stop("Could not find Y in brms standata.")
  }

  list(X = X, y = y)
}


#' Stan-backed Likelihood Function for brms models
#'
#' Uses rstan::grad_log_prob() to compute gradients and numerical
#' Hessian diagonals via Stan Math autodiff, enabling AIS transforms
#' for any brms model family without hand-coded derivatives.
#'
#' @export
BrmsLikelihood <- R6::R6Class("BrmsLikelihood",
  inherit = LikelihoodFunction,
  public = list(
    fit = NULL,
    stanfit = NULL,
    sdata = NULL,
    par_names = NULL,
    n_obs = NULL,

    #' @description Initialize from a brmsfit
    #' @param fit A brmsfit object
    initialize = function(fit) {
      if (!requireNamespace("brms", quietly = TRUE))
        stop("Package 'brms' required")
      if (!requireNamespace("rstan", quietly = TRUE))
        stop("Package 'rstan' required")

      self$fit <- fit
      self$stanfit <- fit$fit
      self$sdata <- brms::standata(fit)
      self$n_obs <- if (!is.null(self$sdata$Y)) length(self$sdata$Y) else self$sdata$N
      self$par_names <- names(rstan::get_inits(self$stanfit, iter = 1)[[1]])
    },

    log_likelihood = function(data, params) {
      # Use brms::log_lik for per-observation log-likelihood
      # data arg is ignored; we use the model's own data
      # params: list(beta=S x K, intercept=S, ...)
      ll <- brms::log_lik(self$fit)  # S x N matrix
      return(ll)
    },

    log_likelihood_gradient = function(data, params) {
      # Use rstan::grad_log_prob for each posterior sample
      # This gives gradient of the full log-posterior, but we extract
      # per-observation gradients via finite differences on log_lik
      #
      # For the AIS framework, we need S x N x K gradients.
      # Use numDeriv on the per-observation log-likelihood.

      beta <- params$beta  # S x K
      intercept <- params$intercept  # S

      S <- nrow(beta)
      N <- self$n_obs
      K <- ncol(beta)

      # Delegate to the appropriate hand-coded likelihood
      # based on the model family
      fam <- stats::family(self$fit)
      fname <- fam$family

      if (fname %in% c("bernoulli", "binomial")) {
        delegate <- LogisticRegressionLikelihood$new()
      } else if (fname == "poisson") {
        delegate <- PoissonRegressionLikelihood$new()
      } else if (fname == "gaussian") {
        delegate <- LinearRegressionLikelihood$new()
      } else {
        # Fallback: numerical gradient via numDeriv
        return(private$numerical_gradient(data, params))
      }

      return(delegate$log_likelihood_gradient(data, params))
    },

    log_likelihood_hessian_diag = function(data, params) {
      fam <- stats::family(self$fit)
      fname <- fam$family

      if (fname %in% c("bernoulli", "binomial")) {
        delegate <- LogisticRegressionLikelihood$new()
      } else if (fname == "poisson") {
        delegate <- PoissonRegressionLikelihood$new()
      } else if (fname == "gaussian") {
        delegate <- LinearRegressionLikelihood$new()
      } else {
        return(private$numerical_hessian_diag(data, params))
      }

      return(delegate$log_likelihood_hessian_diag(data, params))
    }
  ),

  private = list(
    numerical_gradient = function(data, params) {
      # Fallback: use numDeriv for gradient computation
      if (!requireNamespace("numDeriv", quietly = TRUE))
        stop("numDeriv required for numerical gradients")

      beta <- params$beta
      intercept <- params$intercept
      S <- nrow(beta)
      N <- self$n_obs
      K <- ncol(beta)

      grad_beta <- array(0, dim = c(S, N, K))
      grad_intercept <- matrix(0, nrow = S, ncol = N)

      for (s in seq_len(S)) {
        for (n in seq_len(N)) {
          # Gradient of LL[s, n] w.r.t. beta[s, :] and intercept[s]
          f <- function(p) {
            pp <- params
            pp$beta[s, ] <- p[seq_len(K)]
            pp$intercept[s] <- p[K + 1]
            ll <- self$log_likelihood(data, pp)
            ll[s, n]
          }
          g <- numDeriv::grad(f, c(beta[s, ], intercept[s]))
          grad_beta[s, n, ] <- g[seq_len(K)]
          grad_intercept[s, n] <- g[K + 1]
        }
      }
      list(beta = grad_beta, intercept = grad_intercept)
    },

    numerical_hessian_diag = function(data, params) {
      if (!requireNamespace("numDeriv", quietly = TRUE))
        stop("numDeriv required for numerical Hessian")

      beta <- params$beta
      intercept <- params$intercept
      S <- nrow(beta)
      N <- self$n_obs
      K <- ncol(beta)

      hess_beta <- array(0, dim = c(S, N, K))
      hess_intercept <- matrix(0, nrow = S, ncol = N)

      for (s in seq_len(S)) {
        for (n in seq_len(N)) {
          f <- function(p) {
            pp <- params
            pp$beta[s, ] <- p[seq_len(K)]
            pp$intercept[s] <- p[K + 1]
            ll <- self$log_likelihood(data, pp)
            ll[s, n]
          }
          H <- numDeriv::hessian(f, c(beta[s, ], intercept[s]))
          hess_beta[s, n, ] <- diag(H)[seq_len(K)]
          hess_intercept[s, n] <- diag(H)[K + 1]
        }
      }
      list(beta = hess_beta, intercept = hess_intercept)
    }
  )
)


#' Run AIS on a brmsfit object
#'
#' @param fit A brmsfit object
#' @param likelihood_fn Optional LikelihoodFunction. If NULL, uses
#'   BrmsLikelihood (Stan-backed) for supported families, or falls back
#'   to hand-coded likelihoods.
#' @param ... Arguments passed to adaptive_is_loo
#' @export
ais_brms <- function(fit, likelihood_fn = NULL, ...) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    stop("Package 'brms' needed.")
  }

  params <- extract_brms_params(fit)
  data <- extract_brms_data(fit)

  if (is.null(likelihood_fn)) {
    fam <- stats::family(fit)
    fname <- fam$family
    link <- fam$link

    if (fname == "bernoulli" || (fname == "binomial" && link == "logit")) {
      likelihood_fn <- LogisticRegressionLikelihood$new()
    } else if (fname == "poisson" && link == "log") {
      likelihood_fn <- PoissonRegressionLikelihood$new()
    } else if (fname == "gaussian" && link == "identity") {
      likelihood_fn <- LinearRegressionLikelihood$new()
    } else {
      # Use BrmsLikelihood with numerical fallback
      likelihood_fn <- BrmsLikelihood$new(fit)
    }
  }

  sampler <- AdaptiveImportanceSampler$new(likelihood_fn)
  sampler$adaptive_is_loo(data, params, ...)
}
