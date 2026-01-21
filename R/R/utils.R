#' Compute log-sum-exp safely
#'
#' @param x Numeric vector or matrix
#' @return Numeric value or vector
logSumExp <- function(x) {
  if (is.matrix(x)) {
    # Column-wise logSumExp
    xmax <- apply(x, 2, max)
    xmax + log(colSums(exp(x - rep(xmax, each = nrow(x)))))
  } else {
    xmax <- max(x)
    xmax + log(sum(exp(x - xmax)))
  }
}

#' Compute entropy of weights
#'
#' @param weights Numeric vector or matrix of weights
#' @return Entropy value
entropy <- function(weights) {
  # Avoid log(0)
  weights <- weights + 1e-12
  if (is.matrix(weights)) {
    -colSums(weights * log(weights))
  } else {
    -sum(weights * log(weights))
  }
}

#' Map a function over a nested list (simple PyTree equivalent)
#'
#' @param f Function to apply
#' @param x List or atomic object
#' @param ... Additional arguments to f
#' @return Transformed list or object
tree_map <- function(f, x, ...) {
  if (is.list(x) && !is.data.frame(x)) {
    lapply(x, function(child) tree_map(f, child, ...))
  } else {
    f(x, ...)
  }
}

