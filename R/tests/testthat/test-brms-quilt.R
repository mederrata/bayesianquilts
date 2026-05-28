context("brms pairing + ensembling")

# Most tests here only exercise the formula + prior generation, which does
# not require running Stan. The end-to-end fit test is gated on brms +
# rstan being installed AND a fast-enough Stan toolchain.

test_that("quilt_brms_formula assembles (1 | g) and (1 | g:t) terms", {
  skip_if_not_installed("brms")
  i <- Interactions$new(list(Dimension$new("g", 3L), Dimension$new("t", 4L)))
  d <- Decomposed$new(i, 1L, name = "beta")
  f <- quilt_brms_formula("y", d)
  s <- deparse1(brms::brmsterms(f)$formula)
  expect_true(grepl("\\(1 \\| g\\)", s))
  expect_true(grepl("\\(1 \\| t\\)", s))
  expect_true(grepl("\\(1 \\| g:t\\)", s))
})

test_that("quilt_brms_formula appends fixed-effect predictors", {
  skip_if_not_installed("brms")
  i <- Interactions$new(list(Dimension$new("g", 3L)))
  d <- Decomposed$new(i, 1L, name = "beta")
  f <- quilt_brms_formula("y", d, predictors = c("x1", "x2"))
  s <- deparse1(brms::brmsterms(f)$formula)
  expect_true(grepl("x1", s))
  expect_true(grepl("x2", s))
  expect_true(grepl("\\(1 \\| g\\)", s))
})

test_that("quilt_brms_priors emits Intercept + per-group sd priors", {
  skip_if_not_installed("brms")
  i <- Interactions$new(list(Dimension$new("g", 3L), Dimension$new("t", 4L)))
  d <- Decomposed$new(i, 1L, name = "beta")
  scales <- d$generalization_preserving_scales(noise_scale = 1.0,
                                                 total_n = 1000)
  pri <- quilt_brms_priors(d, scales)
  expect_true(inherits(pri, "brmsprior"))
  expect_true(any(pri$class == "Intercept"))
  expect_true(any(pri$class == "sd" & pri$group == "g"))
  expect_true(any(pri$class == "sd" & pri$group == "t"))
  expect_true(any(pri$class == "sd" & pri$group == "g:t"))
})

test_that("fit_quilt_brms runs end-to-end on a tiny dataset", {
  skip_if_not_installed("brms")
  skip_if_not_installed("rstan")
  skip_on_cran()
  # Stan compilation is expensive; gate this further with an env var so
  # local devs can opt in but routine R CMD check stays fast.
  skip_if(!nzchar(Sys.getenv("BQ_RUN_BRMS_TESTS")),
          "set BQ_RUN_BRMS_TESTS=1 to run end-to-end brms fits")

  set.seed(0)
  G <- 4L; T_ <- 3L; N <- 80L
  g <- sample(0:(G - 1L), N, TRUE)
  t <- sample(0:(T_ - 1L), N, TRUE)
  alpha_g <- rnorm(G, 0, 1)
  alpha_t <- rnorm(T_, 0, 0.5)
  y <- 2 + alpha_g[g + 1] + alpha_t[t + 1] + rnorm(N, sd = 0.3)
  df <- data.frame(y = y, g = factor(g), t = factor(t))

  fit <- fit_quilt_brms("y", df,
                        interactions = list(Dimension$new("g", G),
                                            Dimension$new("t", T_)),
                        noise_scale = 0.3,
                        family = stats::gaussian(),
                        chains = 1L, iter = 200L, refresh = 0,
                        seed = 1, silent = 2)
  expect_s3_class(fit, "brmsfit")
})

test_that("component_predict + ensemble_components run on a mock brmsfit", {
  skip_if_not_installed("brms")
  skip_if_not_installed("rstan")
  skip_if(!nzchar(Sys.getenv("BQ_RUN_BRMS_TESTS")),
          "set BQ_RUN_BRMS_TESTS=1 to run end-to-end brms fits")

  set.seed(1)
  G <- 3L; N <- 50L
  g <- sample(0:(G - 1L), N, TRUE)
  alpha_g <- rnorm(G, 0, 1)
  y <- 1 + alpha_g[g + 1] + rnorm(N, sd = 0.5)
  df <- data.frame(y = y, g = factor(g))

  i <- Interactions$new(list(Dimension$new("g", G)))
  d <- Decomposed$new(i, 1L, name = "beta")
  fit <- fit_quilt_brms("y", df, interactions = d,
                        noise_scale = 0.5, family = stats::gaussian(),
                        chains = 1L, iter = 200L, refresh = 0,
                        seed = 7, silent = 2)
  contribs <- component_predict(fit, d)
  expect_named(contribs, names(d$.tensor_part_interactions),
               ignore.order = TRUE)
  ens <- ensemble_components(fit, d, method = "uniform")
  expect_equal(sum(ens$weights), 1, tolerance = 1e-12)
  expect_equal(dim(ens$ensemble),
               c(nrow(contribs[[1]]), ncol(contribs[[1]])))
})

test_that("regex_escape handles brms grouping factor separators", {
  expect_equal(regex_escape("g:t"), "g:t")
  expect_equal(regex_escape("a.b"), "a\\.b")
})
