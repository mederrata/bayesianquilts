#' @include decomposition.R brms_quilt.R
NULL

# ============================================================================
# brms wrapper for the renormalization-group lattice / quilted regression
# models.
#
# The Python predictors `LinearBayesianquilt` and `LogisticBayesianquilt`
# decompose a regression model on a binned representation of continuous
# features:
#
#   1. Each continuous covariate is quantile-binned into K_j cells
#      (the lattice).
#   2. The induced binned multi-index defines a multi-way `Interactions`
#      object whose `Decomposed` parts are the additive components of
#      the regression / intercept surface.
#   3. Component priors come from `generalization_preserving_scales`
#      (manuscript-derived: tau_alpha = sigma * sqrt(c/(1-c)) / sqrt(N^alpha)),
#      optionally with a per-component bound and per-cell sample sizes
#      from a `MultiwayContingencyTable`.
#
# This module wraps that pipeline through `brms` / `rstan`. The binned
# factors become brms grouping factors; each non-excluded interaction
# becomes a `(1 | bin_j1:bin_j2:...)` group-level term; component priors
# become `prior(normal(0, tau_alpha), class = "sd", group = ...)`.
#
# For Gaussian outcomes the noise scale used to compute the prior taus is
# estimated from a plug-in OLS residual unless the caller supplies one.
# For Bernoulli outcomes the noise scale defaults to 1 (logit-scale unit).
# ============================================================================


#' Quantile-bin a numeric vector into K cells.
#'
#' Cell indices are returned as integers in `0..(K-1)`, matching the
#' `Dimension` cardinality convention used by `Interactions`. Constant
#' columns and NAs are tolerated (NAs become cell `0`; constant columns
#' degenerate to a single cell).
#'
#' @param x Numeric vector.
#' @param k Integer, number of cells. Quantile breaks are equispaced at
#'   `seq(0, 1, length.out = k + 1)[2:k]`.
#' @return Integer vector of length `length(x)`, values in `0..(k-1)`.
#' @export
quantile_bin <- function(x, k) {
  k <- as.integer(k)
  if (k < 1L) stop("k must be >= 1")
  if (k == 1L) return(integer(length(x)))
  finite <- is.finite(x)
  if (sum(finite) == 0L) return(integer(length(x)))
  if (length(unique(x[finite])) <= 1L) return(integer(length(x)))
  probs <- seq(0, 1, length.out = k + 1L)[c(-1L, -(k + 1L))]
  breaks <- stats::quantile(x[finite], probs = probs, names = FALSE,
                            type = 7L)
  breaks <- sort(unique(breaks))
  # findInterval returns 0..length(breaks), so each value lands in
  # exactly one of the k cells (clipped to [0, k-1]).
  cell <- findInterval(x, breaks, all.inside = FALSE,
                       left.open = FALSE, rightmost.closed = TRUE)
  cell[!is.finite(x)] <- 0L
  pmin(pmax(as.integer(cell), 0L), k - 1L)
}


#' Build an `Interactions` object from quantile-binned numeric features.
#'
#' @param data Data frame containing the numeric `predictors` columns.
#' @param predictors Character vector of column names to bin.
#' @param k Either a single integer (same K for every feature) or a
#'   named integer vector mapping `predictor -> K`.
#' @param max_order Maximum interaction order to retain (default: full
#'   product, i.e. `length(predictors)`).
#' @param exclusions Optional list of character vectors, each naming a
#'   subset of predictors that should be excluded from the decomposition
#'   (passed through to `Interactions$new`).
#' @return A list with components `interactions` (an `Interactions`
#'   object), `binned` (a data frame with one binned-factor column per
#'   predictor, named `<predictor>` -- replacing the original numeric
#'   columns), and `breaks` (a named list of the quantile break vectors
#'   that produced each binning).
#' @export
quilt_lattice_interactions <- function(data, predictors, k = 4L,
                                       max_order = NULL,
                                       exclusions = NULL) {
  if (length(predictors) == 0L)
    stop("predictors must be non-empty")
  if (length(k) == 1L) {
    k <- setNames(rep(as.integer(k), length(predictors)), predictors)
  } else {
    if (is.null(names(k)) || !all(predictors %in% names(k)))
      stop("k must be either a scalar or a named integer vector covering predictors")
    k <- as.integer(k[predictors])
    names(k) <- predictors
  }

  binned <- as.data.frame(data, stringsAsFactors = FALSE)
  breaks_used <- list()
  for (p in predictors) {
    if (!p %in% names(binned))
      stop(sprintf("predictor %s not in data", p))
    x <- as.numeric(binned[[p]])
    binned[[p]] <- quantile_bin(x, k[[p]])
    # Effective cardinality (collapse if constant column produced 1 cell).
    eff_k <- max(1L, length(unique(binned[[p]])))
    k[[p]] <- as.integer(eff_k)
    breaks_used[[p]] <- if (eff_k > 1L) {
      probs <- seq(0, 1, length.out = eff_k + 1L)[c(-1L, -(eff_k + 1L))]
      sort(unique(stats::quantile(x[is.finite(x)], probs = probs,
                                   names = FALSE, type = 7L)))
    } else {
      numeric(0)
    }
  }

  dims <- lapply(predictors, function(p)
    Dimension$new(p, as.integer(k[[p]]))
  )
  ints <- Interactions$new(dimensions = dims, exclusions = exclusions)
  if (!is.null(max_order)) ints <- ints$truncate_to_order(as.integer(max_order))

  list(interactions = ints, binned = binned, breaks = breaks_used, k = k)
}


# Local null-coalescing helper (avoid depending on rlang).
`%||%` <- function(a, b) if (is.null(a)) b else a


#' Fit a quilted lattice regression with brms.
#'
#' Quantile-bins the continuous `predictors`, builds the additive quilt
#' decomposition from the binned multi-index, computes
#' `generalization_preserving_scales` to set component prior sds, and
#' fits the resulting hierarchical model with `brms::brm`.
#'
#' The fitted brmsfit has one group-level random intercept per
#' non-excluded interaction component; the overall fitted value at
#' covariate point `x` is the sum of those component contributions, which
#' is exactly the additive RG-lattice approximation in the manuscript.
#'
#' @param response Character. Name of the outcome column in `data`.
#' @param data Data frame with `response` plus the continuous predictors
#'   named in `predictors`.
#' @param predictors Character vector of continuous predictor names.
#' @param k Either a single integer (same K per predictor) or a named
#'   integer vector. Number of quantile bins per predictor.
#' @param max_order Maximum interaction order to retain. Defaults to
#'   `length(predictors)` (full product).
#' @param exclusions Optional list of character vectors of predictor
#'   names to drop from the decomposition.
#' @param noise_scale Optional plug-in estimate of sigma. If `NULL`,
#'   defaults to `1.0` for non-gaussian families and to `sd(response)`
#'   for `gaussian()`.
#' @param c Effective-df budget (default 0.5).
#' @param per_component If `TRUE`, use the per-component manuscript
#'   bound (`tau ~ sigma / sqrt(p * N^alpha)`) rather than the per-
#'   parameter bound.
#' @param use_contingency_counts If `TRUE`, derive per-cell sample sizes
#'   from the actual data via a `MultiwayContingencyTable`; otherwise
#'   the uniform-cells assumption is used (default `TRUE`).
#' @param family A brms family. Defaults to `brms::gaussian()`.
#' @param scales Optional named numeric vector of component prior sds to
#'   override `generalization_preserving_scales`.
#' @param ... Forwarded to `brms::brm` (e.g. `chains`, `iter`, `seed`,
#'   `refresh`).
#' @return A list with `fit` (a `brmsfit`), `interactions` (the
#'   `Interactions` object used), `binned` (binned data frame fed to
#'   brms), `breaks` (per-predictor quantile breaks), and `scales` (the
#'   tau_alpha values that drove the priors).
#' @export
fit_quilt_lattice_brms <- function(response, data, predictors,
                                   k = 4L, max_order = NULL,
                                   exclusions = NULL,
                                   noise_scale = NULL,
                                   c = 0.5,
                                   per_component = TRUE,
                                   use_contingency_counts = TRUE,
                                   family = NULL,
                                   scales = NULL, ...) {
  .require_brms()
  if (is.null(family)) family <- brms::gaussian()
  if (!response %in% names(data))
    stop(sprintf("response %s not in data", response))

  lattice <- quilt_lattice_interactions(
    data, predictors = predictors, k = k,
    max_order = max_order, exclusions = exclusions
  )
  binned <- lattice$binned
  binned[[response]] <- data[[response]]

  # Coerce binned factor columns to character so brms can treat them as
  # categorical grouping factors without ambiguity.
  for (p in predictors) binned[[p]] <- as.character(binned[[p]])

  decomposed <- Decomposed$new(lattice$interactions, param_shape = 1L,
                                name = "beta")

  if (is.null(noise_scale)) {
    fam_name <- if (is.list(family) && !is.null(family$family))
      family$family else as.character(family)[1]
    if (identical(fam_name, "gaussian")) {
      noise_scale <- stats::sd(stats::na.omit(data[[response]]))
      if (!is.finite(noise_scale) || noise_scale <= 0) noise_scale <- 1.0
    } else {
      noise_scale <- 1.0
    }
  }

  ctab <- NULL
  if (use_contingency_counts) {
    ctab <- MultiwayContingencyTable$new(lattice$interactions)
    ctab$fit(binned[, predictors, drop = FALSE])
  }

  if (is.null(scales)) {
    scales <- decomposed$generalization_preserving_scales(
      noise_scale = noise_scale,
      total_n = nrow(data),
      contingency_table = ctab,
      c = c,
      per_component = per_component
    )
  }

  fmla <- quilt_brms_formula(response, decomposed, family = family)
  pri <- quilt_brms_priors(decomposed, scales)
  fit <- brms::brm(formula = fmla, data = binned, prior = pri,
                   family = family, ...)

  list(fit = fit, interactions = lattice$interactions,
       decomposed = decomposed, binned = binned,
       breaks = lattice$breaks, scales = scales,
       noise_scale = noise_scale)
}


#' Apply a fitted quilted lattice model to new data.
#'
#' Re-bins the predictors using the breaks stored from the original fit
#' (so new rows are placed in the same lattice cells), then delegates to
#' `brms::posterior_linpred` / `posterior_predict` via component_predict.
#'
#' @param fit_obj The list returned by `fit_quilt_lattice_brms`.
#' @param newdata A data frame with the same predictor columns.
#' @param method `"linpred"` (default) returns posterior draws of the
#'   linear predictor; `"predict"` returns posterior predictive draws.
#' @param ... Forwarded to brms.
#' @return Matrix of posterior draws (S x N).
#' @export
predict_quilt_lattice <- function(fit_obj, newdata,
                                  method = c("linpred", "predict"), ...) {
  .require_brms()
  method <- match.arg(method)
  predictors <- names(fit_obj$breaks)
  rebinned <- as.data.frame(newdata, stringsAsFactors = FALSE)
  for (p in predictors) {
    br <- fit_obj$breaks[[p]]
    x <- as.numeric(rebinned[[p]])
    if (length(br) == 0L) {
      cell <- integer(length(x))
    } else {
      cell <- findInterval(x, br, all.inside = FALSE,
                           left.open = FALSE, rightmost.closed = TRUE)
      cell[!is.finite(x)] <- 0L
      cell <- pmin(pmax(as.integer(cell), 0L),
                   as.integer(fit_obj$decomposed$.interactions$.intrinsic_shape[
                     match(p, predictors)]) - 1L)
    }
    rebinned[[p]] <- as.character(cell)
  }
  if (method == "linpred") {
    brms::posterior_linpred(fit_obj$fit, newdata = rebinned,
                             allow_new_levels = TRUE, ...)
  } else {
    brms::posterior_predict(fit_obj$fit, newdata = rebinned,
                              allow_new_levels = TRUE, ...)
  }
}
