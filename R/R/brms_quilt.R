#' @include decomposition.R brms_utils.R
NULL

# ============================================================================
# brms pairing for quilted models
#
# Translates a `Decomposed` object (R port of the bayesianquilts additive
# interaction decomposition) into a brms formula + prior list, with prior
# scales set from `generalization_preserving_scales`.
#
# Mapping conventions (matching the manuscript's parameterization):
#   - The order-0 component (intersection of no interaction) -> brms Intercept.
#   - An order-k component {d1, ..., dk} -> brms group-level term
#         (1 | d1:d2:...:dk)
#   - Each group-level term gets a `sd` prior `normal(0, tau_alpha)` where
#     tau_alpha is the generalization-preserving scale for component alpha.
#   - Optional fixed-effect predictors are appended to the formula RHS.
# ============================================================================


.require_brms <- function() {
  if (!requireNamespace("brms", quietly = TRUE))
    stop("Package 'brms' is required for this function; install it first.")
}


#' Build a brms formula from a Decomposed quilt structure
#'
#' Returns a `brms::bf` formula with one group-level term per non-excluded
#' interaction component plus the optional fixed-effect predictors.
#'
#' @param response The name of the response variable (character).
#' @param decomposed A `Decomposed` object describing the interaction
#'   components.
#' @param predictors Optional character vector of additional fixed-effect
#'   predictors to include as `+ pred1 + pred2`.
#' @param family Optional brms family (passed straight through to `bf`).
#'   Default `NULL` lets brms decide.
#' @param interaction_sep Separator used inside group-level terms (default
#'   `":"` matches brms's grouping factor convention).
#' @return A `brmsformula` object.
#' @export
quilt_brms_formula <- function(response, decomposed, predictors = NULL,
                                family = NULL, interaction_sep = ":") {
  .require_brms()
  if (!inherits(decomposed, "Decomposed"))
    stop("decomposed must be a Decomposed object")

  rhs_terms <- character(0)
  if (length(predictors) > 0) rhs_terms <- c(rhs_terms, predictors)

  for (name in names(decomposed$.tensor_part_interactions)) {
    vars <- decomposed$.tensor_part_interactions[[name]]
    if (length(vars) == 0L) next   # order-0 is the intercept; brms adds it
    rhs_terms <- c(
      rhs_terms,
      sprintf("(1 | %s)", paste(vars, collapse = interaction_sep))
    )
  }

  rhs <- if (length(rhs_terms) == 0L) "1" else paste(rhs_terms, collapse = " + ")
  fmla_str <- paste(response, "~", rhs)
  fmla <- stats::as.formula(fmla_str)
  if (is.null(family)) brms::bf(fmla) else brms::bf(fmla, family = family)
}


#' Build brms priors from a Decomposed quilt structure
#'
#' For the order-0 component, sets `prior(normal(0, tau0), class = "Intercept")`.
#' For each higher-order component `(d1, ..., dk)`, sets
#' `prior(normal(0, tau_alpha), class = "sd", group = "d1:...:dk")`.
#'
#' Pass the returned object directly to `brms::brm(..., prior = ...)`.
#'
#' @param decomposed A `Decomposed` object.
#' @param scales A named numeric vector / list of prior scales (typically
#'   produced by `decomposed$generalization_preserving_scales(...)`). Names
#'   must match the keys in `decomposed$.tensor_part_interactions`.
#' @param interaction_sep See `quilt_brms_formula`.
#' @param intercept_scale Optional override for the Intercept prior. If
#'   `NULL`, uses the scale for the order-0 component (or 1.0 if absent).
#' @return A brms prior table (the same kind that `brms::prior` returns).
#' @export
quilt_brms_priors <- function(decomposed, scales,
                              interaction_sep = ":",
                              intercept_scale = NULL) {
  .require_brms()
  if (!inherits(decomposed, "Decomposed"))
    stop("decomposed must be a Decomposed object")

  pri_list <- list()
  for (name in names(decomposed$.tensor_part_interactions)) {
    vars <- decomposed$.tensor_part_interactions[[name]]
    tau <- as.numeric(scales[[name]] %||% scales[[paste0(name, "")]])
    if (length(vars) == 0L) {
      tau0 <- if (!is.null(intercept_scale)) as.numeric(intercept_scale)
              else if (length(tau) == 1 && !is.na(tau)) tau else 1.0
      pri_list[[length(pri_list) + 1L]] <- brms::set_prior(
        sprintf("normal(0, %g)", tau0), class = "Intercept"
      )
    } else {
      group <- paste(vars, collapse = interaction_sep)
      if (is.null(tau) || is.na(tau)) tau <- 1.0
      pri_list[[length(pri_list) + 1L]] <- brms::set_prior(
        sprintf("normal(0, %g)", tau), class = "sd", group = group
      )
    }
  }
  Reduce(`+`, pri_list)
}

# Local null-coalescing helper (avoid depending on rlang).
`%||%` <- function(a, b) if (is.null(a)) b else a


#' Fit a quilted brms model with manuscript-derived prior scales
#'
#' Convenience wrapper that:
#'   1. Builds a `Decomposed` from `interactions` (or accepts an existing one).
#'   2. Computes `generalization_preserving_scales` (or uses caller-provided
#'      scales).
#'   3. Assembles the formula + prior list.
#'   4. Calls `brms::brm(...)`.
#'
#' @param response The response variable name (character).
#' @param data A data frame containing the response, predictors, and one
#'   integer-coded (0..cardinality-1) column per dimension named in the
#'   interactions.
#' @param interactions Either a `Decomposed`, an `Interactions`, or a list
#'   passed to `Interactions$new(...)`.
#' @param predictors Optional character vector of additional fixed-effect
#'   predictors (e.g. continuous covariates).
#' @param noise_scale Plug-in estimate of sigma (passed to the scale formula).
#' @param family A brms family (default `gaussian()`).
#' @param contingency_table Optional MultiwayContingencyTable to derive
#'   per-cell sample sizes from the data instead of the uniform assumption.
#' @param c Effective-df budget (default 0.5).
#' @param per_component Use per-component bound (see Decomposed docs).
#' @param scales Optional named numeric of prior scales to skip the formula
#'   derivation (e.g. for sensitivity analysis).
#' @param ... Passed through to `brms::brm` (e.g. `chains`, `iter`, `seed`).
#' @return A `brmsfit` object.
#' @export
fit_quilt_brms <- function(response, data, interactions,
                            predictors = NULL,
                            noise_scale = 1.0,
                            family = NULL,
                            contingency_table = NULL,
                            c = 0.5, per_component = FALSE,
                            scales = NULL, ...) {
  .require_brms()
  if (inherits(interactions, "Decomposed")) {
    decomposed <- interactions
  } else if (inherits(interactions, "Interactions")) {
    decomposed <- Decomposed$new(interactions, param_shape = 1L)
  } else {
    decomposed <- Decomposed$new(Interactions$new(interactions),
                                  param_shape = 1L)
  }
  if (is.null(scales)) {
    scales <- decomposed$generalization_preserving_scales(
      noise_scale = noise_scale,
      total_n = nrow(data),
      contingency_table = contingency_table,
      c = c,
      per_component = per_component
    )
  }
  fmla <- quilt_brms_formula(response, decomposed, predictors = predictors,
                              family = family)
  pri <- quilt_brms_priors(decomposed, scales)

  brms::brm(formula = fmla, data = data, prior = pri, ...)
}
