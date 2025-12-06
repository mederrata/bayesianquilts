#!/usr/bin/env python3
from collections import defaultdict
from itertools import groupby, product
from operator import itemgetter

import jax.numpy as jnp
import numpy as np
from jax import random
from tqdm import tqdm

from bayesianquilts.stackedtensor import broadcast_tensors
from bayesianquilts.util import CountEncoder


def ravel_multi_index(multi_index, dims):
    """JAX implementation of ravel_multi_index.

    Converts multi-dimensional indices to flat indices.

    Args:
        multi_index: Multi-dimensional indices
        dims: Dimensions of the array

    Returns:
        Flat indices
    """
    strides = jnp.cumprod(jnp.array(dims[::-1]), dtype=jnp.int32)[::-1]
    strides = jnp.concatenate([strides[1:], jnp.array([1], dtype=jnp.int32)])
    multi_index = jnp.asarray(multi_index, dtype=strides.dtype)
    return jnp.sum(multi_index * jnp.expand_dims(strides, 1), axis=0)


def ravel_broadcast_tile(
    tensor, from_shape, to_shape, param_ndims=None, batch_ndims=None
):
    """Unravel, tile to match to_shape, ravel, without raveling.

    Args:
        tensor: The tensor is assumed to have shape:
            batch_shape + [1] + param_shape
        from_shape: shape for unraveled version of tensor
        to_shape: shape to tile to
        param_ndims: Number of parameter dimensions
        batch_ndims: Number of batch dimensions

    Returns:
        Broadcast and tiled tensor
    """
    multiple = int(np.prod(to_shape) / np.prod(from_shape))
    tensor_shape = list(tensor.shape)

    if param_ndims is None:
        param_ndims = 0
        for tdim, todim, fdim in zip(
            reversed(tensor_shape), reversed(to_shape), reversed(from_shape)
        ):
            if tdim == todim and fdim == tdim:
                param_ndims += 1
            else:
                break
    param_dims = tensor_shape[-param_ndims:]

    if batch_ndims is None:
        batch_ndims = 0
        for tdim, todim, fdim in zip(
            tensor_shape[: (-param_ndims - 1)],
            to_shape[: (-param_ndims - 1)],
            from_shape[: (-param_ndims - 1)],
        ):
            if tdim == todim and tdim == fdim:
                batch_ndims += 1
            else:
                break
    batch_dims = tensor_shape[:batch_ndims]

    # Tile along new axis
    tensor_ = jnp.tile(tensor[..., jnp.newaxis], [1] * len(tensor_shape) + [multiple])

    broadcast_dims = [k for k, i in enumerate(from_shape[:(-param_ndims)]) if i == 1]

    # Find contiguous regions of broadcast dimensions
    contiguous = [
        list(map(itemgetter(1), g))
        for k, g in groupby(enumerate(broadcast_dims), lambda i_x: i_x[0] - i_x[1])
    ]

    # Transpose to put new dimension in right place
    tensor_ = jnp.transpose(
        tensor_,
        (
            list(range(batch_ndims + 1))
            + [len(tensor_shape)]
            + list(range(batch_ndims + 1, len(tensor_shape)))
        ),
    )

    for chunk in contiguous:
        # Insert each chunk into the right slots
        lower = chunk[0]
        upper = chunk[-1] + 1
        prior_dims = to_shape[batch_ndims:lower]
        chunk_dims = from_shape[lower:upper]
        chunk_dims_to = to_shape[lower:upper]
        post_dims = from_shape[upper:(-param_ndims)]

        _temp_shape = list(tensor_.shape)
        extra_dims = _temp_shape[batch_ndims + 1]
        dims_to_insert = int(np.prod(chunk_dims_to))

        tensor_ = jnp.reshape(
            tensor_,
            (
                (batch_dims if len(batch_dims) > 0 else [1])
                + [int(np.prod(prior_dims))]
                + [int(np.prod(post_dims))]
                + [dims_to_insert]
                + [int(extra_dims / dims_to_insert)]
                + [int(np.prod(param_dims))]
            ),
        )
        tensor_ = jnp.transpose(tensor_, (0, 1, 3, 2, 4, 5))
        _temp_shape = list(tensor_.shape)
        tensor_ = jnp.reshape(
            tensor_,
            (
                (batch_dims if len(batch_dims) > 0 else [1])
                + [int(np.prod(_temp_shape[1:4]))]
                + [int(np.prod(_temp_shape[4:5]))]
                + param_dims
            ),
        )

    _temp_shape = list(tensor_.shape)
    tensor_ = jnp.reshape(
        tensor_,
        batch_dims
        + [int(np.prod(_temp_shape[batch_ndims:(-param_ndims)]))]
        + param_dims,
    )
    return tensor_


class Interactions(object):
    """Manages interaction structure for decomposed parameters.

    Handles multi-way interactions between categorical variables with
    optional exclusions.

    Attributes:
        _dimensions: List of (name, cardinality) tuples
        _intrinsic_shape: Shape of the interaction tensor
        _exclusions: List of excluded interaction sets
    """
    _interaction_list = []

    def __init__(self, dimensions=None, exclusions=None) -> None:
        super().__init__()
        exclusions = [] if exclusions is None else exclusions
        dimensions = [] if dimensions is None else dimensions

        self._dimensions = dimensions
        self._intrinsic_shape = []

        for x in self._dimensions:
            try:
                self._intrinsic_shape += [len(x[1])]
            except TypeError:
                self._intrinsic_shape += [x[1]]

        self._exclusions = [set(s) for s in exclusions]
        if len(self._intrinsic_shape) == 0:
            self._intrinsic_shape = [1]

    def shape(self):
        """Get the shape of the interaction tensor."""
        return self._intrinsic_shape

    def rank(self):
        """Get the rank (number of dimensions) of the interaction."""
        return len(self.shape())

    def __print__(self):
        """Print interaction details."""
        print(f"shape: {self._intrinsic_shape}")
        print(f"dimensions: {self._dimensions}")
        print(f"exclusions: {self._exclusions}")

    def __str__(self):
        """String representation of the interaction."""
        out = f"Interaction dimensions: {self._dimensions}"
        return out

    def retrieve_indices(self, data):
        """Retrieve interaction indices from data dictionary.

        Args:
            data: Dictionary mapping dimension names to indices

        Returns:
            Concatenated indices for all dimensions
        """
        if len(self._dimensions) == 0:
            return jnp.array([0], dtype=jnp.int64)
        indices = [jnp.asarray(data[k[0]], dtype=jnp.int64) for k in self._dimensions]
        return jnp.concatenate(indices, axis=-1)

    def truncate_to_order(self, max_order):
        """Limit interaction to maximum order.

        Args:
            max_order: Maximum number of variables in an interaction

        Returns:
            New Interactions object with order <= max_order
        """
        if len(self._dimensions) == 0:
            return self

        onehot = list(
            filter(
                lambda x: sum(x) > max_order,
                product([0, 1], repeat=len(self._dimensions)),
            )
        )
        exclusions = self._exclusions
        hot = [
            set(d[0] for i, d in zip(ind, self._dimensions) if i == 1)
            for ind in onehot
        ]
        exclusions += hot
        return Interactions(
            self._dimensions,
            exclusions=exclusions,
        )

    def exclude(self, exclusions):
        """Add exclusions to the interaction.

        Args:
            exclusions: List of sets of variable names to exclude

        Returns:
            New Interactions object with additional exclusions
        """
        _exclusions = self._exclusions + [set(s) for s in exclusions]
        exclusions = []
        [exclusions.append(x) for x in _exclusions if x not in exclusions]
        return Interactions(self._dimensions, exclusions=exclusions)

    def __add__(self, other):
        """Combine two interactions.

        Args:
            other: Another Interactions object

        Returns:
            New Interactions object with combined dimensions
        """
        if other is None:
            return self
        exclusions = other._exclusions + self._exclusions
        dimensions = self._dimensions + other._dimensions
        res = []
        [res.append(x) for x in dimensions if x not in res]
        return Interactions(dimensions=res, exclusions=exclusions)


class Decomposed(object):
    """Decomposed parameter representation for hierarchical models.

    Implements additive parameter decomposition across interaction dimensions,
    enabling hierarchical modeling with automatic regularization through the
    decomposition structure.

    The decomposition represents a parameter tensor as:
        θ = Σ θ_interaction

    where each θ_interaction corresponds to a different subset of variables.

    Attributes:
        _interactions: Interactions object defining the structure
        _param_shape: Shape of parameters at each interaction level
        _intrinsic_shape: Full shape (interaction_shape + param_shape)
        _tensor_parts: Dictionary of decomposed parameter tensors
        _tensor_part_interactions: Mapping of tensor names to interactions
        _tensor_part_shapes: Shapes of each tensor part
        scales: Scaling factors for each component
        labels: Human-readable labels for components
    """

    _param_tensors = {}
    _intrinsic_shape = None

    def __init__(
        self,
        interactions,
        param_shape=None,
        default_val=None,
        implicit=False,
        dtype=jnp.float32,
        post_fn=None,
        name="",
        **kwargs,
    ) -> None:
        super().__init__()
        assert isinstance(
            interactions, Interactions
        ), "Instantiation requires a parameterization"

        self._interactions = interactions
        self._default_val = default_val
        self._interaction_shape = interactions.shape()
        self._dtype = dtype
        self._implicit = implicit
        self._name = name
        self._post_fn = post_fn

        if param_shape is None:
            param_shape = [1]
        if len(param_shape) > 5:
            raise NotImplementedError("Param dimensions > 5 are not supported")
        self._param_shape = param_shape
        self._intrinsic_shape = self._interaction_shape + param_shape

        (
            self._tensor_parts,
            self._tensor_part_interactions,
            self._tensor_part_shapes,
        ) = self.generate_tensors()

        self.scales = defaultdict(lambda: 1)
        self.labels = self.generate_labels()

    def generate_labels(self):
        """Generate human-readable labels for each tensor component.

        Returns:
            Dictionary mapping tensor names to lists of labels
        """
        dimension_dict = dict(self._interactions._dimensions)
        _dimension_labels = [
            [f"{j}={t}" for t in dimension_dict[j]]
            if isinstance(dimension_dict[j], list)
            else [f"{j}={t}" for t in range(dimension_dict[j])]
            for j in [x[0] for x in self._interactions._dimensions]
        ]
        _dimension_labels = [" & ".join(t) for t in product(*_dimension_labels)]
        labels = defaultdict(lambda: _dimension_labels)

        for k, v in self._tensor_part_interactions.items():
            if len(v) == 0:
                continue
            dimension_labels = [
                [f"{j}={t}" for t in dimension_dict[j]]
                if isinstance(dimension_dict[j], list)
                else [f"{j}={t}" for t in range(dimension_dict[j])]
                for j in v
            ]
            dimension_labels = [" & ".join(t) for t in product(*dimension_labels)]
            labels[k] = dimension_labels
        return labels

    def generate_tensors(self, batch_shape=None, target=None, skip=None, dtype=None):
        """Generate parameter tensors for the parameter decomposition.

        Handles decomposition of parameters into additive components corresponding
        to different interaction levels.

        Args:
            batch_shape: Optional batch dimensions
            target: Optional target value to decompose
            skip: List of tensor names to skip
            dtype: Data type (defaults to self._dtype)

        Returns:
            Tuple of (tensors, tensor_names, tensor_shapes)
        """
        if skip is None:
            skip = []
        tensors = {}
        tensor_names = {}
        dtype = dtype if dtype is not None else self._dtype
        batch_shape = [] if batch_shape is None else batch_shape
        batch_ndims = len(batch_shape)
        tensor_shapes = {}

        # Handle empty interaction case
        if len(self._interaction_shape) == 0:
            tensor_names[self._name] = ()
            tensor_shapes[self._name] = self._param_shape
            tensors[self._name] = jnp.zeros(self._param_shape, dtype)
            if target is not None:
                tensors[self._name] += target
            return tensors, tensor_names, tensor_shapes

        # Initialize residual
        if target is None:
            residual = jnp.zeros(batch_shape + self.shape(), dtype)
        else:
            residual = jnp.asarray(target, dtype) + jnp.zeros(
                batch_shape + self.shape(), dtype
            )

        # Generate tensors for each interaction
        for n_tuple in product([0, 1], repeat=self._interactions.rank()):
            interaction = tuple(
                [
                    t
                    for j, t in enumerate(self._interactions._dimensions)
                    if n_tuple[j] == 1
                ]
            )
            interaction_vars = tuple([x[0] for x in interaction])

            # Skip excluded interactions
            if set(interaction_vars) in self._interactions._exclusions:
                continue

            interaction_name = "_".join(interaction_vars)
            tensor_name = self._name + "__" + interaction_name

            if tensor_name in skip:
                continue

            # Compute interaction shape
            interaction_shape = (
                np.array(self._interactions.shape()) ** np.array(n_tuple)
            ).tolist()

            tensor_shape = interaction_shape
            if batch_shape is not None:
                tensor_shape = batch_shape + tensor_shape
            tensor_shape += self._param_shape

            # Set the tensor value by averaging over marginalized dimensions
            tensor_names[self._name + "__" + interaction_name] = interaction_vars
            value = residual
            for ax, flag in enumerate(n_tuple):
                if flag == 0:
                    value = jnp.mean(
                        value, axis=(batch_ndims + ax), keepdims=True
                    )

            # Update residual
            residual = residual + (-1.0 * value)

            t_shape = value.shape[batch_ndims:]
            tensor_shapes[self._name + "__" + interaction_name] = t_shape
            interaction_cats = np.prod(t_shape[: (-len(self._param_shape))])

            # Reshape to raveled form
            value = jnp.reshape(
                value,
                batch_shape + [interaction_cats] + self._param_shape,
            )

            # Handle implicit coding (drop first category)
            if self._implicit and (interaction_cats > 1):
                if len(self._param_shape) == 1:
                    value = value[..., 1:, :]
                elif len(self._param_shape) == 2:
                    value = value[..., 1:, :, :]
                elif len(self._param_shape) == 3:
                    value = value[..., 1:, :, :, :]
                elif len(self._param_shape) == 4:
                    value = value[..., 1:, :, :, :, :]
                elif len(self._param_shape) == 5:
                    value = value[..., 1:, :, :, :, :, :]

            tensors[tensor_name] = value

        return tensors, tensor_names, tensor_shapes

    def set_params(self, tensors):
        """Set parameter tensors.

        Args:
            tensors: Dictionary of parameter tensors
        """
        for k in self._tensor_parts.keys():
            if k in tensors:
                self._tensor_parts[k] = tensors[k]

    def set_scales(self, scales):
        """Set scaling factors for components.

        Args:
            scales: Dictionary of scaling factors
        """
        for k in scales.keys():
            self.scales[k] = scales[k]

    def __add__(self, x):
        """Add to the constituted parameter."""
        if isinstance(x, jnp.ndarray):
            return x + self.sum_parts(self._tensor_parts, unravel=True)
        return self.constitute().__add__(x)

    def __radd__(self, x):
        """Right addition."""
        return self.__add__(x)

    def __mul__(self, x):
        """Multiply the constituted parameter."""
        return self.constitute().__mul__(x)

    def shape(self):
        """Get the intrinsic shape of the parameter."""
        return self._intrinsic_shape

    def constitute(self, tensors=None):
        """Alias for sum_parts with unraveling.

        Args:
            tensors: Optional tensor dictionary

        Returns:
            Constituted parameter tensor
        """
        return self.sum_parts(tensors, unravel=True)

    def sum_parts(self, tensors=None, unravel=False, dtype=None):
        """Sum all parameter components.

        Args:
            tensors: Dictionary of tensors to sum (defaults to self._tensor_parts)
            unravel: Whether to reshape to full interaction shape
            dtype: Data type for computation

        Returns:
            Sum of all components
        """
        tensors = self._tensor_parts if tensors is None else tensors
        tensors = {k: v for k, v in tensors.items() if k in self._tensor_parts.keys()}
        dtype = dtype if dtype is not None else self._dtype

        raveled_shape = [np.prod(self._interaction_shape)] + self._param_shape

        # Infer batch shape
        if len(tensors) == 0:
            return jnp.zeros(raveled_shape, dtype)

        batch_shape = list(next(iter(tensors.values())).shape[: (-len(self._param_shape) - 1)])
        partial_sum = jnp.zeros(batch_shape + raveled_shape, dtype)

        # Sum over all tensor parts
        for k, v in tensors.items():
            v = jnp.asarray(v, dtype)
            scale = self.scales[k]
            scale = jnp.asarray(scale, dtype)
            scale = jnp.reshape(
                scale,
                [1] * len(batch_shape)
                + list(scale.shape)
                + [1] * len(self._param_shape),
            )
            v *= scale

            if k not in self._tensor_part_shapes.keys():
                continue

            part_interact_shape = self._tensor_part_shapes[k][
                : (-len(self._param_shape))
            ]
            broadcast_dims = [k for k, i in enumerate(part_interact_shape) if i == 1]
            keep_dims = [k for k, i in enumerate(part_interact_shape) if i != 1]

            if len(keep_dims) == 0:
                partial_sum += v
                continue

            from_shape = batch_shape + self._tensor_part_shapes[k]
            to_shape = batch_shape + self.shape()

            # Handle implicit coding
            if self._implicit and (np.prod(part_interact_shape) > 1):
                v = jnp.pad(
                    v,
                    [[0, 0]] * len(batch_shape)
                    + [[1, 0]]
                    + [[0, 0]] * len(self._param_shape),
                    mode='constant',
                )

            v_ = ravel_broadcast_tile(
                v, from_shape, to_shape, param_ndims=len(self._param_shape)
            )
            partial_sum += v_

        if unravel:
            partial_sum = jnp.reshape(partial_sum, to_shape)

        if self._post_fn is not None:
            partial_sum = self._post_fn(partial_sum)

        return partial_sum

    def lookup(self, interaction_indices, tensors=None, dtype=None):
        """Lookup parameter values at specific interaction indices.

        Args:
            interaction_indices: Indices to lookup
            tensors: Optional tensor dictionary
            dtype: Data type for computation

        Returns:
            Parameter values at specified indices
        """
        return self._lookup_by_parts(interaction_indices, tensors=tensors, dtype=dtype)

    def _lookup_by_parts(self, interaction_indices, tensors=None, dtype=None):
        """Multi-index lookup without summing first.

        Args:
            interaction_indices: Indices to lookup
            tensors: Optional tensor dictionary
            dtype: Data type

        Returns:
            Looked up values
        """
        tensors = self._tensor_parts if tensors is None else tensors
        tensors = {k: v for k, v in tensors.items() if k in self._tensor_parts.keys()}
        dtype = dtype if dtype is not None else self._dtype

        # Handle empty interaction case
        if np.prod(self._interaction_shape) == 1:
            try:
                return tensors[self._name + "__"]
            except KeyError:
                return 0

        interaction_indices = jnp.asarray(interaction_indices)

        if len(tensors) == 0:
            return 0

        batch_shape = list(next(iter(tensors.values())).shape[: (-len(self._param_shape) - 1)])
        cumulative = 0

        for k, tensor in tensors.items():
            if k not in self._tensor_parts.keys():
                continue

            tensor = jnp.asarray(tensor, dtype)
            scale = self.scales[k]
            scale = jnp.asarray(scale, dtype)
            scale = jnp.reshape(
                scale,
                [1] * len(batch_shape)
                + list(scale.shape)
                + [1] * len(self._param_shape),
            )
            tensor *= scale

            if k not in self._tensor_part_shapes.keys():
                continue

            part_interact_shape = self._tensor_part_shapes[k][
                : (-len(self._param_shape))
            ]

            # Handle implicit coding
            if self._implicit and (np.prod(part_interact_shape) > 1):
                _tensor = jnp.pad(
                    tensor,
                    [[0, 0]] * len(batch_shape)
                    + [[1, 0]]
                    + [[0, 0]] * len(self._param_shape),
                    mode='constant',
                )
            else:
                _tensor = tensor

            # Create index selector
            index_select = [1 if k != 1 else 0 for k in part_interact_shape]
            _indices = interaction_indices * jnp.asarray(
                index_select, dtype=interaction_indices.dtype
            )
            _indices = ravel_multi_index(jnp.transpose(_indices), part_interact_shape)

            # Gather values
            batch_ndims = len(batch_shape)
            _tensor = jnp.transpose(
                _tensor,
                list(range(batch_ndims, len(_tensor.shape)))
                + list(range(batch_ndims)),
            )
            _tensor = _tensor[_indices[:, jnp.newaxis]]

            # Move batch dims back to front
            _rank = len(_tensor.shape)
            _tensor = jnp.transpose(
                _tensor,
                list(range(_rank - batch_ndims, _rank))
                + list(range(_rank - batch_ndims)),
            )
            cumulative += _tensor

        if self._post_fn is not None:
            cumulative = self._post_fn(cumulative)

        return cumulative

    def __str__(self):
        """String representation of the decomposed parameter."""
        out = f"Parameter shape: {self._param_shape} \n"
        out += f"{self._interactions} \n"
        out += f"Component tensors: {len(self._tensor_parts.keys())} \n"
        out += f"Effective parameter cardinality: {np.prod(self._intrinsic_shape)} \n"
        out += f"Actual parameter cardinality: {sum([np.prod(t) for t in self._tensor_part_shapes.values()])}\n"
        return out

    def tensor_keys(self):
        """Get sorted list of tensor keys."""
        return sorted(list(self._tensor_parts.keys()))

    def retrieve_indices(self, data):
        """Retrieve interaction indices from data.

        Args:
            data: Data dictionary

        Returns:
            Interaction indices
        """
        return self._interactions.retrieve_indices(data)


class MultiwayContingencyTable(object):
    """Multi-way contingency table for interaction analysis.

    Attributes:
        interaction: Interactions object
        counts: Contingency table counts
    """

    def __init__(self, interaction) -> None:
        self.interaction = interaction

    def fit(self, data_factory, dtype=jnp.int32):
        """Fit contingency table from data.

        Args:
            data_factory: Factory function returning dataset iterator
            dtype: Data type for counts

        Returns:
            Tuple of (counts, labels)
        """
        decomposition = Decomposed(self.interaction, [1])
        n_dims = np.prod(decomposition._interaction_shape)
        counts = jnp.zeros((n_dims,), dtype=dtype)
        dataset = data_factory()
        counter = CountEncoder(list(range(np.prod(decomposition._interaction_shape))))

        for batch in tqdm(iter(dataset)):
            indices = decomposition.retrieve_indices(batch)
            indices = ravel_multi_index(
                jnp.transpose(indices), decomposition._interaction_shape
            )
            counts_ = counter.encode(indices)
            counts += jnp.asarray(counts_[1:], dtype=counts.dtype)

        self.counts = counts
        return counts, decomposition.labels

    def lookup(self, interaction=None):
        """Get corresponding counts for interaction.

        Args:
            interaction: Subset of variables to marginalize to

        Returns:
            Marginalized counts
        """
        if interaction is None:
            return np.sum(self.counts)

        dims = [x[0] for x in self.interaction._dimensions]
        axes = [dims.index(d) for d in interaction]
        otheraxes = [
            d for d in range(len(self.interaction._dimensions)) if d not in axes
        ]
        counts = np.reshape(self.counts, self.interaction._intrinsic_shape)
        counts = np.apply_over_axes(
            np.sum,
            counts,
            [d for d in range(len(self.interaction._intrinsic_shape)) if d not in axes],
        )
        counts = np.transpose(counts, axes + otheraxes)
        counts = np.reshape(counts, counts.shape[: len(axes)])
        return counts


def demo():
    """Demonstration of parameter decomposition functionality."""
    print("Running parameter decomposition demo...")
    print("=" * 60)

    # Example 1: Simple interaction
    print("\nExample 1: Simple MDC interaction")
    interact = Interactions(
        [
            ("MDC", 26), ("HxD1", 3), ("HxD2", 3),
            ("HxD3", 3), ("HxD4", 3), ("HxD5", 3),
            ("HxD6", 3)
        ],
        exclusions=[("HxD1",), ("HxD2",), ("HxD3",), ("HxD4",), ("HxD5",), ()]
    )
    p = Decomposed(interactions=interact, param_shape=[10], name="beta")

    print(f"Interaction shape: {interact.shape()}")
    print(f"Parameter shape: {p.shape()}")
    print(f"Number of tensor parts: {len(p._tensor_parts)}")
    print(f"Tensor keys: {p.tensor_keys()[:3]}...")  # Show first 3

    # Example 2: Lookup with indices
    print("\n" + "=" * 60)
    print("Example 2: Looking up parameter values")
    indices = [
        [21, 1, 1, 1, 1, 2, 1],
        [12, 1, 1, 1, 1, 2, 1],
        [0, 1, 2, 1, 1, 2, 1],
        [13, 1, 2, 1, 1, 2, 1],
        [13, 2, 2, 1, 1, 2, 1]
    ]

    values = p.lookup(indices)
    print(f"Looked up values shape: {values.shape}")
    print(f"Values sample:\n{values[:3]}")

    # Example 3: Batched tensors
    print("\n" + "=" * 60)
    print("Example 3: Batched parameter tensors")
    batch_shape = [4]
    tensors, names, shapes = p.generate_tensors(batch_shape=batch_shape)
    print(f"Generated {len(tensors)} batched tensors")
    print(f"Batch shape: {batch_shape}")
    print(f"Example tensor shape: {list(tensors.values())[0].shape}")

    p.set_params(tensors)
    batched_values = p.lookup(indices)
    print(f"Batched lookup shape: {batched_values.shape}")

    # Example 4: Sum parts
    print("\n" + "=" * 60)
    print("Example 4: Summing parameter parts")
    summed = p.sum_parts(unravel=True)
    print(f"Summed parameter shape: {summed.shape}")
    print(f"Expected shape: {batch_shape + p.shape()}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


def main():
    """Main demonstration function."""
    demo()


if __name__ == "__main__":
    main()
