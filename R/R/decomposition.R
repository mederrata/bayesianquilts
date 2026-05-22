#' @include utils.R
NULL

# ============================================================================
# Quilted-model decomposition machinery (R port of python/bayesianquilts/jax/parameter.py)
#
# This file ports the additive-interaction decomposition that the
# bayesianquilts manuscript uses to organize regularization across multi-way
# categorical interactions. The headline export is
# `generalization_preserving_scales`, which implements the manuscript's
# prior-scale formula for components of any order.
#
# Reference:
#   Chang (2025), "A renormalization-group inspired hierarchical Bayesian
#   framework for piecewise linear regression models".
#
# Scope of the R port:
#   - `Dimension`, `Interactions`, `MultiwayContingencyTable`, and
#     `Decomposed` are ported with the fields and behaviors needed for
#     prior-scale computation and component enumeration.
#   - JAX-specific tensor-lookup plumbing (lookup_flat, sparse indexing,
#     xarray export, NetCDF I/O) is omitted; pair this module with rstan /
#     brms for the actual sampling.
# ============================================================================


#' Dimension of a multi-way interaction
#'
#' Wraps a name + either an integer cardinality or an explicit list of
#' category values.
#'
#' @field name Character. Variable name.
#' @field cardinality Integer. Number of categories.
#' @field values Vector of category values (defaults to 0:(cardinality-1)).
#' @field continuous Logical (default FALSE).
#' @export
Dimension <- R6::R6Class("Dimension",
  public = list(
    name = NULL,
    cardinality = NULL,
    values = NULL,
    continuous = FALSE,

    #' @description Construct a Dimension.
    #' @param name Variable name.
    #' @param cardinality Integer count or a vector of explicit values.
    #' @param continuous Whether the dimension is continuous (default FALSE).
    initialize = function(name, cardinality, continuous = FALSE) {
      self$name <- name
      if (length(cardinality) > 1) {
        self$values <- cardinality
        self$cardinality <- length(cardinality)
      } else {
        self$cardinality <- as.integer(cardinality)
        self$values <- seq.int(0L, self$cardinality - 1L)
      }
      self$continuous <- continuous
    },

    print = function(...) {
      cat(sprintf("Dimension(name=%s, cardinality=%d)\n",
                  self$name, self$cardinality))
    }
  )
)


#' Multi-way interaction structure
#'
#' Holds an ordered list of `Dimension` objects plus an optional list of
#' exclusion sets (each a character vector of dimension names) identifying
#' interaction tuples that should be omitted from the decomposition.
#'
#' @export
Interactions <- R6::R6Class("Interactions",
  public = list(
    .dimensions = NULL,
    .intrinsic_shape = NULL,
    .exclusions = NULL,

    #' @description Construct an Interactions object.
    #' @param dimensions A list of `Dimension` objects or two-element lists
    #'   (name, cardinality).
    #' @param exclusions Optional list of character vectors, each naming a
    #'   subset of dimensions that should be excluded.
    initialize = function(dimensions = NULL, exclusions = NULL) {
      if (is.null(dimensions)) dimensions <- list()
      self$.dimensions <- lapply(dimensions, function(d) {
        if (inherits(d, "Dimension")) d
        else Dimension$new(d[[1]], d[[2]])
      })
      self$.intrinsic_shape <- vapply(self$.dimensions,
                                       function(x) x$cardinality, integer(1))
      if (length(self$.intrinsic_shape) == 0L) self$.intrinsic_shape <- 1L
      self$.exclusions <- if (is.null(exclusions)) list() else
        lapply(exclusions, function(s) sort(unique(as.character(s))))
    },

    shape = function() self$.intrinsic_shape,
    rank = function() length(self$.dimensions),

    #' @description Truncate to interactions of order <= max_order.
    truncate_to_order = function(max_order) {
      n <- length(self$.dimensions)
      if (n == 0) return(self)
      excl <- self$.exclusions
      # Enumerate one-hot inclusion vectors with sum > max_order
      grid <- as.matrix(expand.grid(rep(list(c(0L, 1L)), n)))
      sums <- rowSums(grid)
      keep <- grid[sums > max_order, , drop = FALSE]
      for (i in seq_len(nrow(keep))) {
        excl[[length(excl) + 1]] <- sort(vapply(
          which(keep[i, ] == 1L),
          function(j) self$.dimensions[[j]]$name, character(1)
        ))
      }
      # Deduplicate exclusions
      excl_unique <- list()
      seen <- character(0)
      for (e in excl) {
        key <- paste(e, collapse = "|")
        if (!(key %in% seen)) {
          seen <- c(seen, key)
          excl_unique[[length(excl_unique) + 1]] <- e
        }
      }
      Interactions$new(self$.dimensions, exclusions = excl_unique)
    },

    #' @description Add additional exclusions (returns a new Interactions).
    exclude = function(exclusions) {
      all_excl <- c(self$.exclusions,
                    lapply(exclusions, function(s) sort(unique(as.character(s)))))
      seen <- character(0); out <- list()
      for (e in all_excl) {
        key <- paste(e, collapse = "|")
        if (!(key %in% seen)) { seen <- c(seen, key); out[[length(out)+1]] <- e }
      }
      Interactions$new(self$.dimensions, exclusions = out)
    },

    print = function(...) {
      cat(sprintf("Interactions(dims=%s, shape=%s, n_exclusions=%d)\n",
                  paste(vapply(self$.dimensions, function(d) d$name, character(1)),
                        collapse = ","),
                  paste(self$.intrinsic_shape, collapse = "x"),
                  length(self$.exclusions)))
    }
  )
)


#' Decomposed parameter representation
#'
#' Implements the additive decomposition theta = sum_alpha theta_alpha where
#' alpha indexes a subset of the interaction dimensions. Each component
#' tensor has a shape determined by the included dimensions plus the
#' per-cell parameter shape.
#'
#' The R port stores per-component shapes and interaction memberships and
#' computes prior scales via `generalization_preserving_scales`. The
#' tensor-construction path (`generate_tensors`) is included for
#' completeness; the JAX-specific lookup path is omitted.
#'
#' @export
Decomposed <- R6::R6Class("Decomposed",
  public = list(
    .interactions = NULL,
    .param_shape = NULL,
    .interaction_shape = NULL,
    .intrinsic_shape = NULL,
    .implicit = FALSE,
    .name = "",
    .tensor_part_interactions = NULL,   # named list of character vectors
    .tensor_part_shapes = NULL,         # named list of integer vectors
    scales = NULL,                       # named list of doubles

    #' @description Construct a Decomposed object.
    #' @param interactions An `Interactions` object or a list passed to
    #'   `Interactions$new(...)`.
    #' @param param_shape Integer vector giving the per-cell parameter shape.
    #'   Defaults to `c(1L)` (scalar parameter per cell).
    #' @param implicit If TRUE, drop the first level of each interaction
    #'   dimension (matching the Python `implicit` coding).
    #' @param name Name prefix used for tensor part names.
    initialize = function(interactions, param_shape = NULL,
                          implicit = FALSE, name = "") {
      if (!inherits(interactions, "Interactions"))
        interactions <- Interactions$new(interactions)
      self$.interactions <- interactions
      if (is.null(param_shape)) param_shape <- 1L
      if (length(param_shape) > 5L)
        stop("Param dimensions > 5 are not supported")
      self$.param_shape <- as.integer(param_shape)
      self$.interaction_shape <- interactions$shape()
      self$.intrinsic_shape <- c(self$.interaction_shape, self$.param_shape)
      self$.implicit <- isTRUE(implicit)
      self$.name <- name

      info <- self$.enumerate_components()
      self$.tensor_part_interactions <- info$interactions
      self$.tensor_part_shapes <- info$shapes
      self$scales <- setNames(rep(1.0, length(info$interactions)),
                              names(info$interactions))
    },

    #' @description Enumerate every non-excluded interaction subset.
    .enumerate_components = function() {
      dims <- self$.interactions$.dimensions
      excl <- lapply(self$.interactions$.exclusions, function(s) sort(s))
      rank <- length(dims)
      ints <- list(); shapes <- list()

      if (rank == 0L) {
        nm <- self$.name
        ints[[nm]] <- character(0)
        shapes[[nm]] <- self$.param_shape
        return(list(interactions = ints, shapes = shapes))
      }

      grid <- as.matrix(expand.grid(rep(list(c(0L, 1L)), rank)))
      for (i in seq_len(nrow(grid))) {
        n_tuple <- grid[i, ]
        active <- which(n_tuple == 1L)
        interaction_vars <- if (length(active) == 0) character(0) else
          vapply(active, function(j) dims[[j]]$name, character(1))
        key <- paste(sort(interaction_vars), collapse = "|")
        excl_keys <- vapply(excl, function(s) paste(sort(s), collapse = "|"),
                            character(1))
        if (key %in% excl_keys) next
        interaction_name <- paste(interaction_vars, collapse = "_")
        tensor_name <- paste0(self$.name, "__", interaction_name)

        # Component shape: along each dim, cardinality if active else 1
        shape <- as.integer(self$.interactions$.intrinsic_shape) ^ n_tuple
        # Append param_shape
        shape <- c(as.integer(shape), self$.param_shape)
        # Apply implicit coding (drop first category if more than one)
        if (self$.implicit) {
          # Reduce each active dim by 1, but only if its cardinality > 1
          for (j in seq_along(n_tuple)) {
            if (n_tuple[j] == 1L && shape[j] > 1L) shape[j] <- shape[j] - 1L
          }
        }
        ints[[tensor_name]] <- interaction_vars
        shapes[[tensor_name]] <- shape
      }
      list(interactions = ints, shapes = shapes)
    },

    #' @description Compute generalization-preserving prior scales (manuscript).
    #'
    #' Two modes:
    #'   `per_component = FALSE` (default): per-parameter bound
    #'       tau^alpha = sigma / sqrt(N^alpha)
    #'   `per_component = TRUE`: per-component bound
    #'       tau^alpha = sigma / sqrt(p * N^alpha)
    #' where N^alpha is the average per-cell sample size for component alpha
    #' (uniform unless a `contingency_table` is supplied) and `p` is the
    #' per-cell parameter count.
    #'
    #' @param noise_scale Estimated noise standard deviation sigma.
    #' @param total_n Total sample size (used when contingency_table is NULL).
    #' @param contingency_table Optional `MultiwayContingencyTable` for
    #'   actual per-cell counts.
    #' @param c Maximum effective degrees-of-freedom budget (default 0.5).
    #' @param per_component See above.
    #' @return Named list of doubles, one per component.
    generalization_preserving_scales = function(noise_scale = 1.0,
                                                total_n = NULL,
                                                contingency_table = NULL,
                                                c = 0.5,
                                                per_component = FALSE) {
      if (is.null(total_n) && is.null(contingency_table))
        stop("Must provide either total_n or contingency_table")

      scale_factor <- sqrt(c / (1 - c))
      p <- as.integer(prod(self$.param_shape))
      if (length(p) == 0L) p <- 1L

      out <- list()
      for (name in names(self$.tensor_part_interactions)) {
        interaction_vars <- self$.tensor_part_interactions[[name]]
        shape <- self$.tensor_part_shapes[[name]]
        n_param_dims <- length(self$.param_shape)
        interaction_shape <- shape[seq_len(length(shape) - n_param_dims)]

        if (!is.null(contingency_table)) {
          if (length(interaction_vars) == 0L) {
            n_local <- contingency_table$lookup(NULL)
          } else {
            counts <- contingency_table$lookup(interaction_vars)
            n_local <- mean(as.numeric(counts))
          }
        } else {
          n_cells <- if (length(interaction_shape) == 0L) 1
                     else prod(as.numeric(interaction_shape))
          n_local <- total_n / max(n_cells, 1)
        }
        effective_n <- if (per_component) p * n_local else n_local
        tau <- scale_factor * noise_scale / sqrt(max(effective_n, 1))
        out[[name]] <- as.numeric(tau)
      }
      out
    },

    #' @description Return the interaction order (rank) of a component name.
    component_order = function(component_name) {
      length(self$.tensor_part_interactions[[component_name]])
    },

    #' @description Return all component names at a given interaction order.
    components_at_order = function(order) {
      nms <- names(self$.tensor_part_interactions)
      keep <- vapply(nms, function(n)
        length(self$.tensor_part_interactions[[n]]) == order, logical(1))
      nms[keep]
    },

    #' @description Maximum interaction order present.
    max_order = function() {
      if (length(self$.tensor_part_interactions) == 0L) return(0L)
      max(vapply(self$.tensor_part_interactions, length, integer(1)))
    },

    #' @description Set scales from a named list.
    set_scales = function(scales) {
      for (k in names(scales)) self$scales[[k]] <- scales[[k]]
      invisible(self)
    }
  )
)


#' Multi-way contingency table
#'
#' Counts per cell of an `Interactions` table, used by
#' `Decomposed$generalization_preserving_scales` to derive prior scales
#' from actual data marginals rather than a uniform assumption.
#'
#' @export
MultiwayContingencyTable <- R6::R6Class("MultiwayContingencyTable",
  public = list(
    interaction = NULL,
    counts = NULL,     # array shaped self$interaction$shape()

    #' @description Construct a contingency table for an interaction.
    initialize = function(interaction) {
      if (!inherits(interaction, "Interactions"))
        stop("interaction must be an Interactions object")
      self$interaction <- interaction
    },

    #' @description Fit counts from a data frame (or list of data frames).
    #'
    #' Each named column must match a dimension name. Values are interpreted
    #' as integer category indices in `0..(cardinality - 1)`.
    #'
    #' @param data Data frame or list of data frames (chunks).
    fit = function(data) {
      shape <- self$interaction$.intrinsic_shape
      n_cells <- if (length(shape) == 0L) 1L else as.integer(prod(shape))
      counts_lin <- integer(n_cells)
      chunks <- if (is.data.frame(data)) list(data) else
                if (is.list(data) && !is.data.frame(data[[1]])) data else list(data)
      dim_names <- vapply(self$interaction$.dimensions,
                          function(d) d$name, character(1))
      for (chunk in chunks) {
        if (length(dim_names) == 0L) {
          counts_lin[1] <- counts_lin[1] + nrow(chunk)
          next
        }
        idxs <- lapply(dim_names, function(n) as.integer(chunk[[n]]))
        # Convert (i1, i2, ...) (zero-indexed) to a single linear index
        # using R's column-major convention: lin = i1 + s1*(i2 + s2*(...))
        lin <- idxs[[1]]
        stride <- shape[1]
        if (length(idxs) >= 2L) {
          for (k in 2:length(idxs)) {
            lin <- lin + stride * idxs[[k]]
            stride <- stride * shape[k]
          }
        }
        lin <- lin + 1L   # to R's 1-indexed
        counts_lin <- counts_lin + tabulate(lin, nbins = n_cells)
      }
      self$counts <- if (length(shape) == 0L) counts_lin
                     else array(counts_lin, dim = shape)
      invisible(self)
    },

    #' @description Marginal counts for a subset of dimensions.
    #'
    #' Returns the total count when `interaction` is NULL, otherwise the
    #' counts table summed over all non-included dimensions.
    #'
    #' @param interaction NULL, or a character vector of dimension names.
    lookup = function(interaction = NULL) {
      if (is.null(self$counts))
        stop("Call $fit(data) first")
      if (is.null(interaction)) return(sum(self$counts))
      dim_names <- vapply(self$interaction$.dimensions,
                          function(d) d$name, character(1))
      keep_axes <- match(interaction, dim_names)
      if (anyNA(keep_axes))
        stop("Unknown dimension(s) in interaction: ",
             paste(interaction[is.na(keep_axes)], collapse = ", "))
      sum_axes <- setdiff(seq_along(dim_names), keep_axes)
      if (length(sum_axes) == 0L) return(self$counts)
      out <- apply(self$counts, keep_axes, sum)
      out
    }
  )
)


#' Convenience wrapper: compute manuscript prior scales for a quilted model.
#'
#' Equivalent to `Decomposed$new(interactions, param_shape)$generalization_preserving_scales(...)`
#' but with the most common arguments inlined for users who don't otherwise
#' need a Decomposed instance.
#'
#' @inheritParams Decomposed
#' @param noise_scale Estimated noise standard deviation sigma.
#' @param total_n Total sample size N.
#' @param contingency_table Optional MultiwayContingencyTable.
#' @param c Effective-df budget (default 0.5).
#' @param per_component See Decomposed$generalization_preserving_scales.
#' @return Named list of prior scales, one per non-excluded component.
#' @export
quilt_prior_scales <- function(interactions, param_shape = 1L,
                               noise_scale = 1.0, total_n = NULL,
                               contingency_table = NULL,
                               c = 0.5, per_component = FALSE) {
  d <- Decomposed$new(interactions = interactions, param_shape = param_shape)
  d$generalization_preserving_scales(
    noise_scale = noise_scale,
    total_n = total_n,
    contingency_table = contingency_table,
    c = c,
    per_component = per_component
  )
}
