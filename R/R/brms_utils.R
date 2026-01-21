#' Extract parameters from a brmsfit object for AIS
#'
#' @param fit A brmsfit object
#' @return List of parameters (beta, intercept) suitable for AIS likelihoods
#' @export
extract_brms_params <- function(fit) {
  if (!requireNamespace("brms", quietly = TRUE)) {
    stop("Package 'brms' needed for this function to work.")
  }
  
  # Get all posterior samples
  samples <- brms::as_draws_matrix(fit)
  
  # Check parameter names
  pnames <- colnames(samples)
  
  # Intercept
  # Usually "b_Intercept"
  if ("b_Intercept" %in% pnames) {
    intercept <- samples[, "b_Intercept"]
  } else {
    warning("No 'b_Intercept' found, using 0")
    intercept <- rep(0, nrow(samples))
  }
  
  # Betas
  # All columns starting with "b_" excluding "b_Intercept"
  beta_cols <- grep("^b_", pnames, value = TRUE)
  beta_cols <- setdiff(beta_cols, "b_Intercept")
  
  if (length(beta_cols) > 0) {
    beta <- samples[, beta_cols, drop = FALSE]
  } else {
    beta <- matrix(0, nrow=nrow(samples), ncol=0)
  }
  
  list(beta = beta, intercept = intercept)
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
  
  # brms usually stores predictors in X (matrix) and response in Y (vector/array)
  if (!is.null(sdata$X)) {
    X <- sdata$X
  } else {
    # Maybe only intercept? Or non-linear?
    # Stop if complex
    # Create dummy X if empty
    X <- matrix(0, nrow=length(sdata$Y), ncol=0)
  }
  
  if (!is.null(sdata$Y)) {
    y <- sdata$Y
  } else {
    stop("Could not find Y in brms standata.")
  }
  
  list(X = X, y = y)
}

#' Run AIS on a brmsfit object
#'
#' @param fit A brmsfit object
#' @param likelihood_fn Optional LikelihoodFunction (if NULL, inferred from family)
#' @param ... Arguments passed to adaptive_is_loo
#' @export
ais_brms <- function(fit, likelihood_fn=NULL, ...) {
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
       # Assume logistic regression structure
       # Check if y is binary? brms binomial might imply trials?
       # My LogisticRegressionLikelihood assumes binary y.
       # Need to check.
       likelihood_fn <- LogisticRegressionLikelihood$new()
    } else if (fname == "poisson" && link == "log") {
       likelihood_fn <- PoissonRegressionLikelihood$new()
    } else {
       stop(sprintf("Automatic likelihood inference for family '%s' not supported yet. Please provide 'likelihood_fn' explicitly.", fname))
    }
  }
  
  sampler <- AdaptiveImportanceSampler$new(likelihood_fn)
  sampler$adaptive_is_loo(data, params, ...)
}
