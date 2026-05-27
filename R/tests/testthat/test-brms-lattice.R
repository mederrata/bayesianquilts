context("brms lattice wrapper")

test_that("quantile_bin handles edge cases", {
  expect_equal(quantile_bin(numeric(0), 4), integer(0))
  # NAs are placed in cell 0; finite values land in valid cells.
  cells <- quantile_bin(c(1, NA, 2), 3)
  expect_equal(length(cells), 3L)
  expect_equal(cells[2], 0L)
  expect_true(all(cells %in% 0L:2L))
  # Constant vector -> all in cell 0
  expect_true(all(quantile_bin(rep(2.5, 10), 5) == 0L))
  # k=1 -> all in cell 0
  expect_true(all(quantile_bin(rnorm(100), 1) == 0L))
})

test_that("quantile_bin produces approximately balanced cells", {
  set.seed(0)
  x <- rnorm(1000)
  cells <- quantile_bin(x, 5)
  expect_true(all(cells %in% 0L:4L))
  counts <- tabulate(cells + 1L, nbins = 5L)
  # Each cell should have roughly 200; allow factor-of-2 wobble.
  expect_true(all(counts > 100 & counts < 400))
})

test_that("quilt_lattice_interactions builds Interactions with right cardinalities", {
  set.seed(1)
  df <- data.frame(x1 = rnorm(200), x2 = rnorm(200), x3 = rnorm(200))
  lat <- quilt_lattice_interactions(df, c("x1", "x2", "x3"), k = 3L)
  expect_s3_class(lat$interactions, "Interactions")
  expect_equal(as.integer(lat$interactions$shape()), c(3L, 3L, 3L))
  expect_equal(lat$interactions$rank(), 3L)
  expect_true(all(c("x1", "x2", "x3") %in% names(lat$binned)))
  # Binned columns are integers in 0..2
  for (p in c("x1", "x2", "x3")) {
    expect_true(all(lat$binned[[p]] %in% 0L:2L))
  }
})

test_that("quilt_lattice_interactions honours per-feature k and max_order", {
  set.seed(2)
  df <- data.frame(a = rnorm(100), b = rnorm(100))
  lat <- quilt_lattice_interactions(
    df, c("a", "b"), k = c(a = 2L, b = 4L), max_order = 1L
  )
  expect_equal(as.integer(lat$interactions$shape()), c(2L, 4L))
  # max_order = 1 means the {a, b} 2-way is excluded.
  excl_keys <- vapply(lat$interactions$.exclusions,
                      function(s) paste(sort(s), collapse = "|"),
                      character(1))
  expect_true("a|b" %in% excl_keys)
})

# brms-dependent end-to-end fit
test_that("fit_quilt_lattice_brms runs end-to-end on a tiny dataset", {
  skip_if_not_installed("brms")
  skip_if_not_installed("rstan")
  skip_if(!nzchar(Sys.getenv("BQ_RUN_BRMS_TESTS")),
          "set BQ_RUN_BRMS_TESTS=1 to run end-to-end brms fits")

  set.seed(0)
  N <- 200
  x1 <- rnorm(N); x2 <- rnorm(N)
  y <- 1.5 * x1 - 0.5 * x2 + 0.3 * x1 * x2 + rnorm(N, sd = 0.4)
  df <- data.frame(y = y, x1 = x1, x2 = x2)

  out <- fit_quilt_lattice_brms(
    "y", df, predictors = c("x1", "x2"), k = 4L,
    family = brms::gaussian(),
    chains = 1L, iter = 300L, refresh = 0,
    seed = 1, silent = 2
  )
  expect_s3_class(out$fit, "brmsfit")
  expect_true(all(unlist(out$scales) > 0))
  # Predict on the training data through the same binning
  pp <- predict_quilt_lattice(out, df, method = "linpred")
  expect_equal(ncol(pp), N)
})
