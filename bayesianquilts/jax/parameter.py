#!/usr/bin/env python3
from itertools import product

import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.internal.backend.numpy import gather_nd
from tqdm import tqdm


class Interactions(object):
    _interaction_list = []

    def __init__(
            self,
            dimensions,
            exclusions=None) -> None:
        super().__init__()
        self._dimensions = dimensions
        self._intrinsic_shape = [
            x[1] for x in self._dimensions
        ]
        self._exclusions = [set(s) for s in exclusions]

    def shape(self):
        return self._intrinsic_shape

    def rank(self):
        return len(self.shape())

    def exclude(self, interaction):
        pass


class Decomposed(object):
    _param_tensors = {}
    _intrinsic_shape = None

    def __init__(
            self,
            interactions,
            param_shape=[],
            default_val=None,
            dtype=jnp.float32,
            name="",
            *args,
            **kwargs) -> None:

        super().__init__()
        assert isinstance(
            interactions, Interactions
        ), "Instantiation requires a parameterization"

        self._interactions = interactions
        self._default_val = default_val
        self._intrinsic_shape = interactions.shape() + param_shape
        self._dtype = dtype
        self._name = name
        self._param_shape = param_shape
        self._param_tensors, self._param_interactions = self.generate_tensors()
        self._param_shapes = {
            k: v.shape for k, v in self._param_tensors.items()}

    def generate_tensors(self, batch_shape=None, target=None):
        """Generate parameter tensors for the parameter decomposition,
           neatly handling TF's limitation in the maximum number of
           axes (6) for a tensor to be used in most common mathematical operations
           like broadcasting.

        Args:
            batch_shape ([type], optional): [description]. Defaults to None.
            target ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        tensors = {}
        tensor_names = {}
        batch_shape = [] if batch_shape is None else batch_shape
        batch_ndims = len(batch_shape)
        if target is None:
            residual = jnp.array(
                np.zeros(batch_shape + self.shape(), dtype=self._dtype))
        else:
            residual = (
                target.astype(self._dtype) +
                jnp.array(
                    np.zeros(
                        batch_shape + self.shape(),
                        dtype=self._dtype))
            )
        for n_tuple in product([0, 1], repeat=self._interactions.rank()):
            interaction = tuple(
                [t for j, t in enumerate(
                    self._interactions._dimensions) if n_tuple[j] == 1])
            interaction_vars = tuple([x[0] for x in interaction])
            if set(interaction_vars) in self._interactions._exclusions:
                continue
            interaction_name = "_".join(interaction_vars)
            interaction_shape = (
                jnp.array(self._interactions.shape())**jnp.array(n_tuple)
            ).tolist()

            tensor_shape = interaction_shape
            if batch_shape is not None:
                tensor_shape = batch_shape + tensor_shape
            tensor_shape += self._param_shape

            # Set the tensor value
            tensor_names[self._name +
                         "__" + interaction_name] = interaction_vars
            value = residual
            for ax, flag in enumerate(n_tuple):
                if flag == 0:
                    value = jnp.mean(value, axis=(
                        batch_ndims+ax), keepdims=True)
            tensors[self._name + "__" + interaction_name] = value
            residual = residual - 1.0*value

        return tensors, tensor_names

    def set_params(self, tensors):
        for k in self._param_tensors.keys():
            self._param_tensors[k] = tensors[k]

    def __add__(self, x):
        return self.constitute().__add__(x)

    def __radd__(self, x):
        return self.__add__(x)

    def __mul__(self, x):
        return self.constitute().__mul__(x)

    def shape(self):
        return self._intrinsic_shape

    def constitute(self, tensors=None):
        tensors = self._param_tensors if tensors is None else tensors
        partial_sum = jnp.array(
            np.zeros(self.shape()),
            dtype=self._dtype)
        for _, t in self._param_tensors.items():
            partial_sum += t
        return partial_sum

    def tensor_keys(self):
        return sorted(list(self._param_tensors.keys()))

    def query(self, interaction_indices):
        _global = self.constitute()
        batch_ndims = _global.ndim - len(self.shape())
        # stick batch axes on the  end
        rank = _global.ndim

        _global = jnp.transpose(
            _global,
            list(range(batch_ndims, rank)) + list(range(batch_ndims))
        )
        localized = gather_nd(_global, interaction_indices)

        local_rank = localized.ndim
        localized = jnp.transpose(
            localized,
            list(range(local_rank-batch_ndims, local_rank)) +
            list(range(local_rank-batch_ndims))
        )
        return localized

    def shape(self):
        return self._intrinsic_shape


def main():
    interact = Interactions(
        [
            ("MDC", 26), ("HxD1", 3), ("HxD2", 3),
            ("HxD3", 3), ("HxD4", 3), ("HxD5", 3),
            ("HxD6", 3)],
        # exclusions=[("Dx",),(),("Dx","Tx","Hx")],
        exclusions=[("HxD1",), ("HxD2",), ("HxD3",), ("HxD4",), ("HxD5",), ()]
    )
    p = Decomposed(interactions=interact, param_shape=[1000], name="beta")
    indices = [
        [21, 1, 1, 1, 1, 2, 1],
        [12, 1, 1, 1, 1, 2, 1],
        [0, 1, 2, 1, 1, 2,  1],
        [13, 1, 2, 1, 1, 2,  1],
        [13, 2, 2, 1, 1, 2,  1]]
    q = p.query(indices)

    t = p.generate_tensors(batch_shape=[4])
    p.set_params(t[0])
    r = p.query(indices)
    pass

def ravel_multi_index(multi_index, dims):
    strides = jnp.cumprod(dims, exclusive=True, reverse=True)
    multi_index = jnp.astype(multi_index, strides.dtype)
    return jnp.sum(multi_index * jnp.expand_dims(strides, 1), axis=0)


def ravel_broadcast_tile(
    tensor, from_shape, to_shape, param_ndims=None, batch_ndims=None
):
    """Unravel, tile to match to_shape, ravel,
    without raveling.

    Args:
        tensor ([type]): The tensor is assumed to have shape:
            batch_shape + [1] + param_shape
        from_shape ([type]): shape for unraveled version of tensor
        to_shape ([type]): shape to tile to
    """
    multiple = int(np.prod(to_shape) / np.prod(from_shape))
    tensor_shape = tensor.shape.as_list()
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
    tensor_ = jnp.tile(tensor[..., jnp.newaxis], [1] * len(tensor_shape) + [multiple])

    broadcast_dims = [k for k, i in enumerate(from_shape[:(-param_ndims)]) if i == 1]

    # we need to re-arrange tensor_, putting things in the right place
    # traverse broadcast_dims, looking for continuous regions

    contiguous = [
        list(map(itemgetter(1), g))
        for k, g in groupby(enumerate(broadcast_dims), lambda i_x: i_x[0] - i_x[1])
    ]

    # the last dimension is now the product of all of the missing
    # dimensions in broadcast_dims. Let's stick this dimension into
    # position batch_ndims + 1, so that we have two index dimensions

    tensor_ = jnp.transpose(
        tensor_,
        (
            list(range(batch_ndims + 1))
            + [len(tensor_shape)]
            + list(range(batch_ndims + 1, len(tensor_shape)))
        ),
    )

    for chunk in contiguous:
        # insert each chunk into the right slots
        lower = chunk[0]
        upper = chunk[-1] + 1
        # prior_dims are dimensions already in from_shape
        # that are before this chunk
        prior_dims = to_shape[batch_ndims:lower]
        chunk_dims = from_shape[lower:upper]
        chunk_dims_to = to_shape[lower:upper]
        post_dims = from_shape[upper:(-param_ndims)]
        # post_dims are dimensions already in from_shape
        # that are after this chunk

        # reshape tensor_ to
        # (raveled) batch_dims, (raveled) prior_dims, (raveled)post_dims,
        # (raveled) chunk_dims
        #   (raveled) extra_dims, (raveled) param_dims

        _temp_shape = tensor_.shape.as_list()
        extra_dims = _temp_shape[batch_ndims + 1]

        dims_to_insert = int(np.prod(chunk_dims_to))

        tensor_ = tf.reshape(
            tensor_,
            (
                (batch_dims if len(batch_dims) > 0 else [1])
                + [int(np.prod(prior_dims))]
                + [int(np.prod(post_dims))]
                + [dims_to_insert]
                + [int(extra_dims / dims_to_insert)]
                + [np.prod(param_dims)]
            ),
        )
        tensor_ = jnp.transpose(tensor_, (0, 1, 3, 2, 4, 5))
        _temp_shape = tensor_.shape.as_list()
        tensor_ = jnp.reshape(
            tensor_,
            (
                (batch_dims if len(batch_dims) > 0 else [1])
                + [int(np.prod(_temp_shape[1:4]))]
                + [int(np.prod(_temp_shape[4:5]))]
                + param_dims
            ),
        )

    _temp_shape = tensor_.shape.as_list()
    tensor_ = jnp.reshape(
        tensor_,
        batch_dims
        + [int(np.prod(_temp_shape[batch_ndims:(-param_ndims)]))]
        + param_dims,
    )
    return tensor_


class MultiwayContingencyTable(object):
    def __init__(self, interaction) -> None:
        self.interaction = interaction

    def fit(self, data_factory, dtype=jnp.int32):
        decomposition = Decomposed(self.interaction, [1])
        n_dims = np.prod(decomposition._interaction_shape)
        counts = jnp.zeros((n_dims), dtype=dtype)
        dataset = data_factory()
        counter = CountEncoder(list(range(np.prod(decomposition._interaction_shape))))
        for batch in tqdm(iter(dataset)):
            indices = decomposition.retrieve_indices(batch)
            indices = ravel_multi_index(
                jnp.transpose(indices), decomposition._interaction_shape
            )
            counts_ = counter.encode(indices)
            counts += jnp.astype(counts_[1:], counts.dtype)
        self.counts = counts
        return counts, decomposition.labels

    def lookup(self, interaction=None):
        """
        Get the correstponding counts
        """
        # first, make sure interaction is a subset of self.interaction
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


if __name__ == "__main__":
    main()
