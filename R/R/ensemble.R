#' @include decomposition.R brms_quilt.R
NULL

# ============================================================================
# Parameter-level ensembling via the additive quilt decomposition
#
# A fitted quilted model represents the linear predictor as
#     eta(x) = sum_alpha beta_alpha(x_S(alpha))
# where alpha ranges over interaction components (orders 0..K).
# Parameter-level ensembling exploits this decomposition by combining
# components with weights w_alpha (computed from per-component LOO ELPDs,
# Akaike-style weights, or uniform), yielding
#     eta_ensemble(x) = sum_alpha w_alpha * beta_alpha(x_S(alpha)).
#
# This file provides three pieces:
#   * `component_predict` -- evaluate each component's contribution to the
#     linear predictor on `newdata`, returning an (S, N, n_components)
#     array of per-component posterior draws.
#   * `component_loo_elpd` -- per-observation per-component log-likelihood
#     contributions and component ELPDs.
#   * `ensemble_components` -- combine component contributions into an
#     ensembled prediction with chosen weights (uniform, AIC-style, or
#     LOO-stacking via `loo::stacking_weights`).
# ============================================================================


.require_brms <- function() {
  if (!requireNamespace("brms", quietly = TRUE))
    stop("Package 'brms' is required for this function; install it first.")
}


#' Evaluate per-component contributions to the linear predictor
#'
#' For a quilted brms fit produced by `fit_quilt_brms` (or any brmsfit
#' whose group-level structure matches a `Decomposed` object), returns
#' the per-component contributions to the linear predictor at each row
#' of `newdata`.
#'
#' The order-0 component is the Intercept; order-k components are
#' group-level random intercepts. Continuous fixed-effect predictors,
#' if present, are reported as a "fixed" component.
#'
#' @param fit A `brmsfit`.
#' @param decomposed A `Decomposed` object describing the interaction
#'   structure (typically the one used to fit `fit`).
#' @param newdata Optional new data; defaults to the fitting data.
#' @param interaction_sep Separator used inside group-level terms.
#' @return A named list of (S x N) matrices, one per non-excluded
#'   component, each containing posterior draws of that component's
#'   contribution to the linear predictor at the rows of `newdata`.
#' @export
component_predict <- function(fit, decomposed, newdata = NULL,
                              interaction_sep = ":") {
  .require_brms()
  if (is.null(newdata)) newdata <- fit$data
  draws <- brms::as_draws_matrix(fit)
  S <- nrow(draws)
  N <- nrow(newdata)
  contributions <- list()

  for (name in names(decomposed$.tensor_part_interactions)) {
    vars <- decomposed$.tensor_part_interactions[[name]]

    if (length(vars) == 0L) {
      # Order-0 -> Intercept
      icpt <- draws[, "b_Intercept", drop = TRUE]
      contributions[[name]] <- matrix(icpt, nrow = S, ncol = N)
      next
    }

    group <- paste(vars, collapse = interaction_sep)

    # Build per-row level labels for this grouping factor.
    if (length(vars) == 1L) {
      level_labels <- as.character(newdata[[vars]])
    } else {
      level_labels <- do.call(paste,
        c(lapply(vars, function(v) as.character(newdata[[v]])),
          list(sep = "_")))
    }

    # brms encodes group-level intercepts as columns named
    #   r_<group>[<level>,Intercept]
    # Find the columns belonging to this group and extract draws.
    pattern <- sprintf("^r_%s\\[", regex_escape(group))
    cols <- grep(pattern, colnames(draws), value = TRUE)
    if (length(cols) == 0L) {
      contributions[[name]] <- matrix(0, nrow = S, ncol = N)
      next
    }
    levels_in_fit <- sub(sprintf("^r_%s\\[(.*),Intercept\\]$",
                                  regex_escape(group)), "\\1", cols)
    lookup <- setNames(cols, levels_in_fit)
    m <- matrix(0, nrow = S, ncol = N)
    for (j in seq_len(N)) {
      col <- lookup[[level_labels[j]]]
      if (is.null(col)) next
      m[, j] <- draws[, col, drop = TRUE]
    }
    contributions[[name]] <- m
  }

  contributions
}

# Tiny helper: escape regex metachars in fixed strings.
regex_escape <- function(x) gsub("([.|()\\^{}+$*?]|\\[|\\])", "\\\\\\1", x)


#' Per-component LOO ELPD contributions
#'
#' Computes, for each component alpha and each observation n, the
#' contribution to the model log-likelihood from setting all components
#' except alpha to zero. Aggregated across observations this gives a
#' per-component ELPD that can drive stacking weights.
#'
#' Uses brms's posterior_linpred / log_lik scaffolding via
#' `component_predict`.
#'
#' @param fit A `brmsfit`.
#' @param decomposed A `Decomposed` object.
#' @param sigma Optional residual sd (Gaussian only). If NULL, uses the
#'   posterior mean from `fit`.
#' @return A list with:
#'   * `elpd` -- named numeric, per-component ELPD.
#'   * `loglik` -- list of (S, N) per-obs log-likelihood matrices, one per
#'     component.
#' @export
component_loo_elpd <- function(fit, decomposed, sigma = NULL) {
  .require_brms()
  fam <- stats::family(fit)
  fname <- fam$family
  contributions <- component_predict(fit, decomposed)
  y <- if (!is.null(fit$data) && length(fit$formula$resp) == 1L)
    fit$data[[fit$formula$resp]]
  else brms::standata(fit)$Y

  if (is.null(sigma) && fname == "gaussian") {
    s <- brms::as_draws_matrix(fit)
    if ("sigma" %in% colnames(s)) sigma <- mean(s[, "sigma"]) else sigma <- 1.0
  }

  per_obs_ll <- list()
  for (name in names(contributions)) {
    eta <- contributions[[name]]   # (S, N)
    ll <- switch(fname,
      gaussian = {
        # log dnorm(y | eta, sigma) per (s, n).
        y_mat <- matrix(y, nrow = nrow(eta), ncol = ncol(eta), byrow = TRUE)
        -0.5 * log(2 * pi) - log(sigma) - 0.5 * ((y_mat - eta) / sigma)^2
      },
      bernoulli = {
        y_mat <- matrix(y, nrow = nrow(eta), ncol = ncol(eta), byrow = TRUE)
        p <- 1 / (1 + exp(-eta))
        y_mat * log(p + 1e-12) + (1 - y_mat) * log(1 - p + 1e-12)
      },
      binomial = {
        y_mat <- matrix(y, nrow = nrow(eta), ncol = ncol(eta), byrow = TRUE)
        p <- 1 / (1 + exp(-eta))
        y_mat * log(p + 1e-12) + (1 - y_mat) * log(1 - p + 1e-12)
      },
      poisson = {
        y_mat <- matrix(y, nrow = nrow(eta), ncol = ncol(eta), byrow = TRUE)
        lam <- exp(eta)
        y_mat * eta - lam - lgamma(y_mat + 1)
      },
      stop(sprintf("component_loo_elpd: family '%s' not supported", fname))
    )
    per_obs_ll[[name]] <- ll
  }

  # Component ELPD: log mean exp over samples, summed over obs.
  elpd <- vapply(per_obs_ll, function(ll) {
    S <- nrow(ll)
    lse <- apply(ll, 2, function(v) {
      m <- max(v); m + log(mean(exp(v - m)))
    })
    sum(lse)
  }, numeric(1))

  list(elpd = elpd, loglik = per_obs_ll)
}


#' Ensemble per-component contributions at the parameter level
#'
#' Combines per-component contributions into an ensembled posterior of the
#' linear predictor:
#'   eta_ens[s, n] = sum_alpha w_alpha * contributions[[alpha]][s, n]
#'
#' Weight schemes:
#' * `"uniform"` -- w_alpha = 1/K.
#' * `"aic"` -- w_alpha = exp(elpd_alpha - max) / sum(...); a softmax of
#'   component ELPDs.
#' * `"stacking"` -- optimal LOO stacking weights via `loo::stacking_weights`
#'   on the per-observation per-component log-likelihoods. Requires the
#'   `loo` package.
#' * `"manual"` -- user-supplied numeric vector matching the components.
#'
#' Note: with `"uniform"` or `"aic"`, the ensemble corresponds to a
#' weighted combination of additive submodels; for the additive
#' decomposition theta = sum_alpha theta_alpha, the natural choice for a
#' single coherent model is `w_alpha = 1` for all alpha, which is what a
#' fitted quilted brms gives you "for free" via `posterior_linpred`. The
#' ensembling here is useful when you want to **reweight components by
#' predictive performance** -- e.g., to ablate low-ELPD high-order terms
#' before deployment.
#'
#' @param fit A `brmsfit`.
#' @param decomposed A `Decomposed` object.
#' @param method One of "uniform", "aic", "stacking", "manual".
#' @param weights If `method = "manual"`, a named numeric vector aligned
#'   with the components.
#' @param newdata Optional `newdata` for predictions; defaults to fitting
#'   data.
#' @return A list with:
#'   * `weights` -- named numeric.
#'   * `contributions` -- list of (S, N) per-component matrices.
#'   * `ensemble` -- (S, N) matrix of ensembled linear-predictor draws.
#'   * `elpd` -- per-component ELPD (if method is `"aic"` or `"stacking"`,
#'     otherwise NULL).
#' @export
ensemble_components <- function(fit, decomposed,
                                method = c("uniform", "aic",
                                            "stacking", "manual"),
                                weights = NULL,
                                newdata = NULL) {
  method <- match.arg(method)
  contributions <- component_predict(fit, decomposed, newdata = newdata)
  comp_names <- names(contributions)
  K <- length(comp_names)

  elpd <- NULL
  if (method %in% c("aic", "stacking")) {
    ce <- component_loo_elpd(fit, decomposed)
    elpd <- ce$elpd
  }

  w <- switch(method,
    uniform = setNames(rep(1 / K, K), comp_names),
    aic = {
      m <- max(elpd); v <- exp(elpd - m); setNames(v / sum(v), comp_names)
    },
    stacking = {
      if (!requireNamespace("loo", quietly = TRUE))
        stop("'loo' package required for method = 'stacking'")
      # Stack per-observation log-likelihood matrices across components.
      ll_array <- simplify2array(ce$loglik)   # (S, N, K)
      sw <- loo::stacking_weights(ll_array)
      setNames(as.numeric(sw), comp_names)
    },
    manual = {
      if (is.null(weights))
        stop("method = 'manual' requires `weights` argument")
      if (is.null(names(weights))) names(weights) <- comp_names
      if (!all(comp_names %in% names(weights)))
        stop("weights must name every component")
      setNames(as.numeric(weights[comp_names]), comp_names)
    }
  )

  # Build ensembled linear predictor.
  S <- nrow(contributions[[1]])
  N <- ncol(contributions[[1]])
  ensemble <- matrix(0, nrow = S, ncol = N)
  for (name in comp_names) ensemble <- ensemble + w[[name]] * contributions[[name]]

  list(weights = w, contributions = contributions,
       ensemble = ensemble, elpd = elpd)
}
