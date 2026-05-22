#' Compute log-sum-exp safely
#'
#' @param x Numeric vector or matrix
#' @return Numeric value or vector
#' @export
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
#' @export
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
#' @export
tree_map <- function(f, x, ...) {
  if (is.list(x) && !is.data.frame(x)) {
    lapply(x, function(child) tree_map(f, child, ...))
  } else {
    f(x, ...)
  }
}

#' Flatten PyTree leaves to a single (S, N, K_total) array.
#'
#' Each leaf is either (S, N) (scalar param) or (S, N, K_i) (vector param).
#' Returns the concatenated (S, N, K_total) array plus a layout descriptor
#' that can be passed to unflatten_pytree_leaves() to recover the structure.
#'
#' @param leaves Named list of arrays, each with shape (S, N) or (S, N, ...)
#' @param current_log_ell Reference (S, N) array used when leaves are absent.
#' @return list(flat = array, layout = list of (name, orig_shape, n_elems))
#' @keywords internal
flatten_pytree_leaves <- function(leaves, current_log_ell) {
  S <- nrow(current_log_ell)
  N <- ncol(current_log_ell)
  layout <- list()
  parts <- list()
  for (name in names(leaves)) {
    leaf <- leaves[[name]]
    d <- dim(leaf)
    if (is.null(d) || length(d) == 2) {
      parts[[length(parts) + 1]] <- array(leaf, dim = c(S, N, 1))
      layout[[length(layout) + 1]] <- list(name = name,
                                            orig_dim = if (is.null(d)) c(S, N) else d,
                                            n_elems = 1L)
    } else {
      trailing <- prod(d[-(1:2)])
      parts[[length(parts) + 1]] <- array(leaf, dim = c(S, N, trailing))
      layout[[length(layout) + 1]] <- list(name = name, orig_dim = d,
                                            n_elems = as.integer(trailing))
    }
  }
  K_total <- sum(vapply(layout, function(x) x$n_elems, integer(1)))
  flat <- array(0, dim = c(S, N, K_total))
  idx <- 1L
  for (i in seq_along(parts)) {
    k_i <- layout[[i]]$n_elems
    flat[, , idx:(idx + k_i - 1)] <- parts[[i]]
    idx <- idx + k_i
  }
  list(flat = flat, layout = layout)
}

#' Inverse of flatten_pytree_leaves.
#' @param flat_arr Array of shape (S, N, K_total).
#' @param layout As returned by flatten_pytree_leaves.
#' @return Named list of arrays matching the original PyTree.
#' @keywords internal
unflatten_pytree_leaves <- function(flat_arr, layout) {
  out <- list()
  idx <- 1L
  for (i in seq_along(layout)) {
    entry <- layout[[i]]
    k_i <- entry$n_elems
    chunk <- flat_arr[, , idx:(idx + k_i - 1), drop = FALSE]
    if (length(entry$orig_dim) == 2) {
      out[[entry$name]] <- matrix(chunk, nrow = dim(flat_arr)[1],
                                  ncol = dim(flat_arr)[2])
    } else {
      out[[entry$name]] <- array(chunk, dim = entry$orig_dim)
    }
    idx <- idx + k_i
  }
  out
}

