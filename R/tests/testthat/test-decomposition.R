context("decomposition + manuscript prior scales")

test_that("Dimension wraps name + cardinality", {
  d <- Dimension$new("group", 5L)
  expect_equal(d$name, "group")
  expect_equal(d$cardinality, 5L)
  expect_equal(d$values, 0:4)

  d2 <- Dimension$new("region", c("east", "west"))
  expect_equal(d2$cardinality, 2L)
  expect_equal(d2$values, c("east", "west"))
})

test_that("Interactions tracks shape, rank, and exclusions", {
  i <- Interactions$new(
    dimensions = list(Dimension$new("g", 3L), Dimension$new("t", 4L))
  )
  expect_equal(i$shape(), c(3L, 4L))
  expect_equal(i$rank(), 2L)
  expect_equal(length(i$.exclusions), 0L)

  i2 <- i$truncate_to_order(1L)
  # Removing the 2-way interaction "g & t"
  excl_keys <- vapply(i2$.exclusions, function(s) paste(s, collapse="|"),
                      character(1))
  expect_true("g|t" %in% excl_keys || "t|g" %in% excl_keys)
})

test_that("Decomposed enumerates 2^d - excluded components", {
  i <- Interactions$new(
    list(Dimension$new("g", 3L), Dimension$new("t", 4L))
  )
  d <- Decomposed$new(interactions = i, param_shape = 1L, name = "beta")
  # 2^2 = 4 components: "" (constant), g, t, g_t
  expect_equal(length(d$.tensor_part_interactions), 4L)
  expect_true("beta__" %in% names(d$.tensor_part_interactions))
  expect_true("beta__g_t" %in% names(d$.tensor_part_interactions))
})

test_that("generalization_preserving_scales matches the manuscript formula", {
  # Single 1-way interaction with cardinality K, total N, scalar param.
  # n_local = N / K, p = 1.
  # tau^constant = sigma * sqrt(c/(1-c)) / sqrt(N)
  # tau^var      = sigma * sqrt(c/(1-c)) / sqrt(N/K)
  K <- 5L; N_total <- 1000; sigma <- 2.0; c <- 0.5
  d <- Decomposed$new(interactions = Interactions$new(list(Dimension$new("g", K))),
                      param_shape = 1L, name = "beta")
  scales <- d$generalization_preserving_scales(noise_scale = sigma,
                                                total_n = N_total, c = c)
  factor <- sqrt(c / (1 - c))
  expect_equal(scales[["beta__"]], sigma * factor / sqrt(N_total),
               tolerance = 1e-12)
  expect_equal(scales[["beta__g"]], sigma * factor / sqrt(N_total / K),
               tolerance = 1e-12)
})

test_that("MultiwayContingencyTable fits counts and marginalizes", {
  d1 <- Dimension$new("g", 3L)
  d2 <- Dimension$new("t", 2L)
  i <- Interactions$new(list(d1, d2))
  mct <- MultiwayContingencyTable$new(i)
  df <- data.frame(
    g = c(0, 0, 1, 1, 1, 2, 2, 2),   # 2x g=0, 3x g=1, 3x g=2
    t = c(0, 1, 0, 0, 1, 0, 1, 1)    # 1x (0,1), 2x (1,0), 1x (1,1), 1x (2,0), 2x (2,1)
  )
  mct$fit(df)
  expect_equal(sum(mct$counts), 8L)
  expect_equal(mct$lookup(NULL), 8L)
  # Marginalize on g
  marg_g <- as.integer(mct$lookup("g"))
  expect_equal(marg_g, c(2L, 3L, 3L))
  # Marginalize on t
  marg_t <- as.integer(mct$lookup("t"))
  expect_equal(marg_t, c(4L, 4L))
})

test_that("quilt_prior_scales convenience wrapper matches Decomposed result", {
  i <- Interactions$new(list(Dimension$new("g", 4L)))
  expected <- Decomposed$new(i, 1L,
                              name = "beta")$generalization_preserving_scales(
    noise_scale = 1.0, total_n = 500)
  observed <- quilt_prior_scales(i, param_shape = 1L,
                                  noise_scale = 1.0, total_n = 500)
  # The convenience wrapper uses empty name prefix; just verify same values.
  expect_equal(sort(unname(unlist(observed))),
               sort(unname(unlist(expected))),
               tolerance = 1e-12)
})

test_that("per_component scaling tightens by sqrt(p)", {
  d <- Decomposed$new(Interactions$new(list(Dimension$new("g", 3L))),
                      param_shape = 4L, name = "beta")
  s_param <- d$generalization_preserving_scales(noise_scale = 1.0,
                                                  total_n = 600,
                                                  per_component = FALSE)
  s_comp <- d$generalization_preserving_scales(noise_scale = 1.0,
                                                 total_n = 600,
                                                 per_component = TRUE)
  expect_equal(s_comp[["beta__g"]], s_param[["beta__g"]] / sqrt(4),
               tolerance = 1e-12)
})
