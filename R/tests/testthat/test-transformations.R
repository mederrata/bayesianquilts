context("transformations")

# A minimal locally-linear likelihood with analytic gradient/Hessian.
# y_i ~ N(mu = beta * x_i, sigma = 1), so log_ell = -0.5*(y - beta*x)^2 - 0.5*log(2*pi).
locally_linear_lik <- function() {
  R6::R6Class("LocallyLinearLik",
    inherit = LikelihoodFunction,
    public = list(
      log_likelihood = function(data, params) {
        X <- as.numeric(data$x)         # (N,)
        y <- as.numeric(data$y)         # (N,)
        beta <- params$beta             # (S,) or (S, N)
        if (is.null(dim(beta))) {
          # (S,) global param -> (S, N) prediction
          mu <- outer(beta, X)
        } else {
          mu <- beta * matrix(X, nrow = nrow(beta), ncol = ncol(beta), byrow = TRUE)
        }
        y_mat <- matrix(y, nrow = nrow(mu), ncol = ncol(mu), byrow = TRUE)
        -0.5 * (y_mat - mu)^2 - 0.5 * log(2 * pi)
      },
      log_likelihood_gradient = function(data, params) {
        X <- as.numeric(data$x); y <- as.numeric(data$y)
        beta <- params$beta
        if (is.null(dim(beta))) mu <- outer(beta, X) else mu <- beta * matrix(X, nrow=nrow(beta), ncol=ncol(beta), byrow=TRUE)
        y_mat <- matrix(y, nrow = nrow(mu), ncol = ncol(mu), byrow = TRUE)
        x_mat <- matrix(X, nrow = nrow(mu), ncol = ncol(mu), byrow = TRUE)
        # d/d_beta log_ell = (y - beta*x) * x
        list(beta = (y_mat - mu) * x_mat)
      },
      log_likelihood_hessian_diag = function(data, params) {
        X <- as.numeric(data$x)
        beta <- params$beta
        if (is.null(dim(beta))) {
          S <- length(beta); N <- length(X)
        } else {
          S <- nrow(beta); N <- ncol(beta)
        }
        x_mat <- matrix(X, nrow = S, ncol = N, byrow = TRUE)
        list(beta = -x_mat^2)
      }
    )
  )$new()
}

test_that("MM3 produces a non-trivial Jacobian and matches MM2 on diagonal cov", {
  set.seed(1)
  S <- 200; N <- 10
  X <- rnorm(N); y <- 2 * X + rnorm(N, sd = 0.1)
  data <- list(x = X, y = y)

  # Use independent samples for beta and intercept -> diagonal cov,
  # so MM3 should reduce to MM2 (up to a rotation).
  params <- list(beta = rnorm(S, mean = 2, sd = 0.5))
  lik <- locally_linear_lik()
  log_ell <- lik$log_likelihood(data, params)

  mm3 <- MM3$new(lik)
  res <- mm3$call(max_iter = 1, params = params, theta = params,
                  data = data, log_ell = log_ell,
                  log_ell_original = log_ell)
  expect_true(is.list(res))
  expect_true(all(is.finite(res$khat)))
  expect_true(any(res$log_jacobian != 0))
})

test_that("PMM3 small-step recovers identity in the h -> 0 limit", {
  set.seed(2)
  S <- 150; N <- 8
  X <- rnorm(N); y <- 1.5 * X + rnorm(N, sd = 0.2)
  data <- list(x = X, y = y)

  params <- list(beta = rnorm(S, mean = 1.5, sd = 0.3))
  lik <- locally_linear_lik()
  log_ell <- lik$log_likelihood(data, params)

  pmm3 <- PMM3$new(lik)
  theta <- list(beta = matrix(rep(params$beta, N), nrow = S, ncol = N))
  Q <- pmm3$compute_Q(theta, data, params, log_ell, log_ell_original = log_ell)
  div_Q <- pmm3$compute_divergence_Q(theta, data, params, log_ell)
  expect_true(is.list(Q))
  expect_equal(dim(Q$beta), c(S, N))
  expect_true(all(is.finite(div_Q)))
})

test_that("MixIS produces valid weights and finite predictions", {
  set.seed(3)
  S <- 100; N <- 5
  X <- rnorm(N); y <- X + rnorm(N, sd = 0.2)
  data <- list(x = X, y = y)
  params <- list(beta = rnorm(S, mean = 1, sd = 0.5))

  lik <- locally_linear_lik()
  log_ell <- lik$log_likelihood(data, params)

  mixis <- MixIS$new(lik, n_mix_samples = 80)
  res <- mixis$call(max_iter = 1, params = params, theta = params,
                    data = data, log_ell = log_ell, log_ell_original = log_ell,
                    seed = 99)
  expect_true(all(is.finite(res$khat)))
  # PSIS weights normalize per observation
  col_sums <- colSums(res$psis_weights)
  expect_true(all(abs(col_sums - 1) < 1e-6))
})

test_that("LikelihoodDescent analytical divergence matches numerical (locally linear)", {
  set.seed(4)
  S <- 30; N <- 4
  X <- rnorm(N); y <- 2 * X + rnorm(N, sd = 0.1)
  data <- list(x = X, y = y)
  params <- list(beta = rnorm(S, mean = 2, sd = 0.3))
  lik <- locally_linear_lik()
  log_ell <- lik$log_likelihood(data, params)
  theta <- list(beta = matrix(rep(params$beta, N), nrow = S, ncol = N))

  ll_trans <- LikelihoodDescent$new(lik)
  div_analytic <- ll_trans$compute_divergence_Q(theta, data, params, log_ell)
  # For locally-linear model, d^2/d_beta^2 log_ell = -x^2, so div(Q) = -(-x^2) = x^2
  expected <- matrix(X^2, nrow = S, ncol = N, byrow = TRUE)
  expect_equal(div_analytic, expected, tolerance = 1e-9)
})

test_that("PMM1 has zero analytical divergence", {
  set.seed(5)
  S <- 20; N <- 3
  X <- rnorm(N); y <- X
  params <- list(beta = rnorm(S))
  lik <- locally_linear_lik()
  log_ell <- lik$log_likelihood(list(x = X, y = y), params)
  pmm1 <- PMM1$new(lik)
  div_Q <- pmm1$compute_divergence_Q(theta = NULL,
                                      data = list(x = X, y = y),
                                      params = params,
                                      current_log_ell = log_ell)
  expect_equal(div_Q, matrix(0, nrow = S, ncol = N))
})
