#!/usr/bin/env python3
from email.policy import default
import re
from collections import Counter, defaultdict
from itertools import product, groupby
from operator import itemgetter
import numpy as np
from tqdm import tqdm

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd
from bayesianquilts.util import CountEncoder

from bayesianquilts.stackedtensor import broadcast_tensors


def tf_ravel_multi_index(multi_index, dims):
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    multi_index = tf.cast(multi_index, strides.dtype)
    return tf.reduce_sum(multi_index * tf.expand_dims(strides, 1), axis=0)


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
    tensor_ = tf.tile(tensor[..., tf.newaxis], [1] * len(tensor_shape) + [multiple])

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

    tensor_ = tf.transpose(
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
        tensor_ = tf.transpose(tensor_, (0, 1, 3, 2, 4, 5))
        _temp_shape = tensor_.shape.as_list()
        tensor_ = tf.reshape(
            tensor_,
            (
                (batch_dims if len(batch_dims) > 0 else [1])
                + [int(np.prod(_temp_shape[1:4]))]
                + [int(np.prod(_temp_shape[4:5]))]
                + param_dims
            ),
        )

    _temp_shape = tensor_.shape.as_list()
    tensor_ = tf.reshape(
        tensor_,
        batch_dims
        + [int(np.prod(_temp_shape[batch_ndims:(-param_ndims)]))]
        + param_dims,
    )
    return tensor_


class Interactions(object):
    _interaction_list = []

    def __init__(self, dimensions, exclusions=None) -> None:
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
        return self._intrinsic_shape

    def rank(self):
        return len(self.shape())

    def __print__(self):
        print(f"shape: {self._intrinsic_shape}")
        print(f"dimensions: {self._dimensions}")
        print(f"exclusions: {self._exclusions}")

    def __str__(self):
        out = f"Interaction dimensions: {self._dimensions}"
        return out

    def retrieve_indices(self, data):
        if len(self._dimensions) == 0:
            return tf.cast([0], tf.int64)
        indices = [tf.cast(data[k[0]], tf.int64) for k in self._dimensions]
        return tf.concat(indices, axis=-1)

    def truncate_to_order(self, max_order):
        if len(self._dimensions) == 0:
            return self
        onehot = list(
            filter(
                lambda x: sum(x) > max_order,
                product([0, 1], repeat=len(self._dimensions)),
            )
        )
        exclusions = []
        hot = [
            set(d[0] for i, d in zip(ind, self._dimensions) if i == 1) for ind in onehot
        ]
        exclusions += hot
        return Interactions(
            self._dimensions,
            exclusions=exclusions,
        )
    
    def __add__(self, other):
        if other is None:
            return self
        exclusions = other.exclusions + self._exclusions
        dimensions = self._dimensions + other._dimensions
        res = []
        [res.append(x) for x in dimensions if x not in res]
        return Interactions(dimensions=res, exclusions=exclusions)



class Decomposed(object):
    """Class for dealing with decomposed parameters

    This class deals with broadcasting and slicing of raveled decomposed
    parameters.

    Attributes
    -----------

    Methods
    -------

    """

    _param_tensors = {}
    _intrinsic_shape = None

    def __init__(
        self,
        interactions,
        param_shape=None,
        default_val=None,
        implicit=True,
        dtype=tf.float32,
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
        # generate a label for each tensor part
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

    def generate_tensors(self, batch_shape=None, target=None, dtype=None):
        """Generate parameter tensors for the parameter decomposition,
           neatly handling TF's limitation in the maximum number of
           axes (6) for a tensor to be used in most common mathematical operations
           like broadcasting.

        Args:
            batch_shape ([list], optional): [description]. Defaults to None.
            target ([list], optional): [description]. Value to target for decomposition.
            flatten_indices (boolean): Whether to flatten index dimensions

        Returns:
            [type]: [description]
        """
        tensors = {}
        tensor_names = {}
        dtype = dtype if dtype is not None else self._dtype
        batch_shape = [] if batch_shape is None else batch_shape
        batch_ndims = len(batch_shape)
        tensor_shapes = {}

        if len(self._interaction_shape) == 0:
            tensor_names[self._name] = ()
            tensor_shapes[self._name] = self._param_shape
            tensors[self._name] = tf.zeros(self._param_shape, dtype)
            if target is not None:
                tensors[self._name] += target
            return tensors, tensor_names, tensor_shapes

        if target is None:
            residual = tf.zeros(batch_shape + self.shape(), dtype)
        else:
            residual = tf.cast(target, dtype) + tf.zeros(
                batch_shape + self.shape(), dtype
            )
        for n_tuple in product([0, 1], repeat=self._interactions.rank()):
            interaction = tuple(
                [
                    t
                    for j, t in enumerate(self._interactions._dimensions)
                    if n_tuple[j] == 1
                ]
            )
            interaction_vars = tuple([x[0] for x in interaction])
            if set(interaction_vars) in self._interactions._exclusions:
                continue
            interaction_name = "_".join(interaction_vars)
            interaction_shape = (
                self._interactions.shape() ** np.array(n_tuple)
            ).tolist()

            tensor_shape = interaction_shape
            if batch_shape is not None:
                tensor_shape = batch_shape + tensor_shape
            tensor_shape += self._param_shape

            # Set the tensor value
            tensor_names[self._name + "__" + interaction_name] = interaction_vars
            value = residual
            for ax, flag in enumerate(n_tuple):
                if flag == 0:
                    value = tf.reduce_mean(
                        value, axis=(batch_ndims + ax), keepdims=True
                    )

            residual = tf.add_n(broadcast_tensors([residual, -1.0 * value]))
            t_shape = value.shape.as_list()[batch_ndims:]
            tensor_shapes[self._name + "__" + interaction_name] = t_shape
            interaction_cats = np.prod(t_shape[: (-len(self._param_shape))])
            value = tf.reshape(
                value,
                batch_shape + [interaction_cats] + self._param_shape,
            )
            value = tf.constant(value)
            if self._implicit and (interaction_cats > 1):
                if len(self._param_shape) == 1:
                    value = value[..., 1:, :]
                elif len(self._param_shape) == 2:
                    value = value[..., 1:, :, :]
                elif len(self._param_shape) == 3:
                    value = value[..., 1:, :, :, :]
                elif len(self._param_shape) == 4:
                    value = (value[..., 1:, :, :, :, :],)
                elif len(self._param_shape) == 5:
                    value = value[..., 1:, :, :, :, :, :]
                elif len(self._param_shape) == 6:
                    value = value[..., 1:, :, :, :, :, :, :]
            tensors[self._name + "__" + interaction_name] = value

        return tensors, tensor_names, tensor_shapes

    def set_params(self, tensors):
        for k in self._param_tensors.keys():
            self._param_tensors[k] = tensors[k]

    def set_scales(self, scales):
        for k in scales.keys():
            self.scales[k] = scales[k]

    def flatten_indices(self, tensors=None):
        param_rank = len(self._param_shape)
        interaction_rank = len(self._interaction_shape)
        for k in tensors.keys():
            rank = tf.rank(tensors[k])
            batch_shape = tensors[k].shape.as_list()[
                : (rank - param_rank - interaction_rank)
            ]
            interaction_shape = self._param_shapes[k][:(-param_rank)]
            tensors[k] = tf.reshape(
                tensors[k],
                batch_shape + [np.prod(interaction_shape)] + self._param_shape,
            )

        return tensors

    def inflate_indices(self, tensors):
        param_rank = len(self._param_shape)
        interaction_rank = len(self._interaction_shape)
        for k in tensors.keys():
            try:
                self._tensor_part_shapes[k]
            except KeyError:
                continue
            rank = len(tensors[k].shape.as_list())
            batch_shape = tensors[k].shape.as_list()[: (rank - param_rank - 1)]
            tensors[k] = tf.reshape(
                tensors[k], batch_shape + self._tensor_part_shapes[k]
            )
        return tensors

    def __add__(self, x):
        if tf.is_tensor(x):
            return tf.add_n(
                broadcast_tensors(
                    [x, self.unravel_tensor(self.sum_parts(self._tensor_parts))]
                )
            )
        return self.constitute().__add__(x)

    def __radd__(self, x):
        return self.__add__(x)

    def __mul__(self, x):
        return self._tensor_parts().__mul__(x)

    def shape(self):
        return self._intrinsic_shape

    def sum_parts(self, tensors=None, unravel=False, dtype=tf.float32):
        tensors = self._tensor_parts if tensors is None else tensors
        raveled_shape = [np.prod(self._interaction_shape)] + self._param_shape
        # infer the batch shape

        batch_shape = tf.nest.flatten(tensors)[0].shape.as_list()[
            : (-len(self._param_shape) - 1)
        ]
        partial_sum = tf.zeros(batch_shape + raveled_shape, dtype)
        #  batch

        # folded
        for k, v in tensors.items():
            v = tf.cast(v, dtype)
            # shuffle axes, placing broadcast axes together
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
            # pad interaction dimension with leading zeros
            if self._implicit and (np.prod(part_interact_shape) > 1):
                v = tf.pad(
                    v,
                    [[0, 0]] * len(batch_shape)
                    + [[1, 0]]
                    + [[0, 0]] * len(self._param_shape),
                    "CONSTANT",
                )
            v_ = ravel_broadcast_tile(
                v, from_shape, to_shape, param_ndims=len(self._param_shape)
            )
            partial_sum += self.scales[k] * v_

        if unravel:
            partial_sum = tf.reshape(partial_sum, to_shape)
        return partial_sum

    def unravel_tensor(self, tensor):
        pass

    def __str__(self):
        out = f"Parameter shape: {self._param_shape} \n"
        out += f"{self._interactions} \n"
        out += f"Component tensors: {len(self._tensor_parts.keys())} \n"
        out += f"Effective parameter cardinality: {np.prod(self._intrinsic_shape)} \n"
        out += f"Actual parameter cardinality: {sum([np.prod(t) for t in self._tensor_part_shapes.values()])}\n"
        return out

    def tensor_keys(self):
        return sorted(list(self._param_tensors.keys()))

    def _lookup_by_parts(self, interaction_indices, tensors=None, dtype=tf.float32):
        """Multi-index lookup without summing

        Args:
            interaction_indices ([type]): [description]
            tensors ([type], optional): [description]. Defaults to None.
        """
        # flatten the indices
        tensors = self._tensor_parts if tensors is None else tensors
        if np.prod(self._interaction_shape) == 1:
            return tensors[self._name + "__"]
        interaction_indices = tf.convert_to_tensor(interaction_indices)

        interaction_shape = tf.convert_to_tensor(
            self._interaction_shape, dtype=interaction_indices.dtype
        )
        batch_shape = tf.nest.flatten(tensors)[0].shape.as_list()[
            : (-len(self._param_shape) - 1)
        ]
        cumulative = 0
        for k, tensor in tensors.items():
            tensor = tf.cast(tensor, dtype)
            if k not in self._tensor_part_shapes.keys():
                continue
            part_interact_shape = self._tensor_part_shapes[k][
                : (-len(self._param_shape))
            ]
            if self._implicit and (np.prod(part_interact_shape) > 1):
                _tensor = tf.pad(
                    tensor,
                    [[0, 0]] * len(batch_shape)
                    + [[1, 0]]
                    + [[0, 0]] * len(self._param_shape),
                    "CONSTANT",
                )
            else:
                _tensor = tensor

            batch_ndims = len(batch_shape)
            """
            # reshape tensor, re-adding the dimensions
            _tensor = tf.reshape(
                _tensor, batch_shape + self._tensor_part_shapes[k]
            )
            new_rank = len(_tensor.shape.as_list())
            
            # transpose the batch dims to the back
            permutation = list(range(batch_ndims, new_rank)) + \
                list(range(batch_ndims))
            _tensor = tf.transpose(_tensor, permutation)
            # gather
            """
            index_select = [1 if k != 1 else 0 for k in part_interact_shape]
            # move batch back up
            _indices = interaction_indices * tf.cast(
                index_select, interaction_indices.dtype
            )
            _indices = tf_ravel_multi_index(tf.transpose(_indices), part_interact_shape)
            _tensor = tf.transpose(
                _tensor,
                list(range(batch_ndims, len(_tensor.shape.as_list())))
                + list(range(batch_ndims)),
            )
            _tensor = tf.gather_nd(_tensor, _indices[:, tf.newaxis])

            # move batch dims back to front
            _rank = len(_tensor.shape.as_list())
            _tensor = tf.transpose(
                _tensor,
                list(range(_rank - batch_ndims, _rank))
                + list(range(_rank - batch_ndims)),
            )
            cumulative += _tensor
            # add to cumulative sum
        return cumulative

    def dot_sum(
        self, interaction_indices, y, tensors=None, axes=[-1], dtype=tf.float32
    ):
        # y has to have a compatible shape
        """Multi-index mult without summing, then sum on axes

        Args:
            interaction_indices ([type]): [description]
            tensors ([type], optional): [description]. Defaults to None.
        """
        tensors = self._tensor_parts if tensors is None else tensors
        if len(self._interaction_shape) == 0:
            return tf.reduce_sum(tensors[self._name] * y, axes)
        # flatten the indices
        interaction_indices = tf.convert_to_tensor(interaction_indices)
        # assert interaction_indices.shape.as_list()[-1] == len(self._interaction_shape)
        tensors = self._tensor_parts if tensors is None else tensors

        interaction_shape = tf.convert_to_tensor(
            self._interaction_shape, dtype=interaction_indices.dtype
        )
        batch_shape = tf.nest.flatten(tensors)[0].shape.as_list()[
            : (-len(self._param_shape) - 1)
        ]
        cumulative = 0
        for k, tensor in tensors.items():
            tensor = tf.cast(tensor, dtype)
            if k not in self._tensor_part_shapes.keys():
                continue
            part_interact_shape = self._tensor_part_shapes[k][
                : (-len(self._param_shape))
            ]
            if self._implicit and (np.prod(part_interact_shape) > 1):
                _tensor = tf.pad(
                    tensor,
                    [[0, 0]] * len(batch_shape)
                    + [[1, 0]]
                    + [[0, 0]] * len(self._param_shape),
                    "CONSTANT",
                )
            else:
                _tensor = tensor

            batch_ndims = len(batch_shape)
            """
            # reshape tensor, re-adding the dimensions
            _tensor = tf.reshape(
                _tensor, batch_shape + self._tensor_part_shapes[k]
            )
            new_rank = len(_tensor.shape.as_list())
            
            # transpose the batch dims to the back
            permutation = list(range(batch_ndims, new_rank)) + \
                list(range(batch_ndims))
            _tensor = tf.transpose(_tensor, permutation)
            # gather
            """
            index_select = [1 if k != 1 else 0 for k in part_interact_shape]
            # move batch back up
            _indices = interaction_indices * tf.cast(
                index_select, interaction_indices.dtype
            )
            _indices = tf_ravel_multi_index(tf.transpose(_indices), part_interact_shape)
            _tensor = tf.transpose(
                _tensor,
                list(range(batch_ndims, len(_tensor.shape.as_list())))
                + list(range(batch_ndims)),
            )
            _tensor = tf.gather_nd(_tensor, _indices[:, tf.newaxis])

            # move batch dims back to front
            _rank = len(_tensor.shape.as_list())
            _tensor = tf.transpose(
                _tensor,
                list(range(_rank - batch_ndims, _rank))
                + list(range(_rank - batch_ndims)),
            )
            part = tf.reduce_sum(_tensor * tf.cast(y, _tensor.dtype), axes)
            cumulative += part
            # add to cumulative sum
        return cumulative

    def lookup(self, interaction_indices, tensors=None, dtype=tf.float32):
        return self._lookup_by_parts(interaction_indices, tensors=tensors, dtype=dtype)

    def _lookup_by_sum(self, interaction_indices, tensors=None, dtype=tf.float32):
        tensors = self._tensor_parts if tensors is None else tensors

        if len(self._interaction_shape) == 0:
            return tensors[self._name]
        # flatten the indices
        interaction_indices = tf.convert_to_tensor(interaction_indices)
        # assert interaction_indices.shape.as_list()[-1] == len(self._interaction_shape)

        interaction_shape = tf.convert_to_tensor(
            self._interaction_shape, dtype=interaction_indices.dtype
        )

        summed = self.sum_parts(tensors, dtype=dtype)

        overall_shape = summed.shape.as_list()
        rank = len(overall_shape)
        batch_ndims = rank - len(self._param_shape) - 1

        new_rank = len(summed.shape.as_list())

        # stick batch axes on the  end
        permutation = list(range(batch_ndims, new_rank)) + list(range(batch_ndims))
        summed = tf.transpose(summed, permutation)

        interaction_indices = tf.transpose(tf.convert_to_tensor(interaction_indices))
        indices = tf_ravel_multi_index(interaction_indices, self._interaction_shape)
        summed = tf.gather_nd(summed, indices[:, tf.newaxis])

        rank = len(summed.shape.as_list())
        # move parameter batch dims back to the front
        permutation = list(range(rank - batch_ndims, rank)) + list(
            range(rank - batch_ndims)
        )
        summed = tf.transpose(summed, permutation)

        return summed

    def _lookup_indices_inflated(self, interaction_indices, deflated_tensors=None):
        summed = self._constitute_inflated(deflated_tensors)
        rank = len(summed.shape.as_list())

        batch_ndims = rank - len(self.shape())

        # stick batch axes on the  end
        permutation = list(range(batch_ndims, rank)) + list(range(batch_ndims))
        summed = tf.transpose(summed, permutation)
        summed = tf.reshape(
            summed,
            [np.prod(summed.shape[: -(len(self._param_shape) + batch_ndims)])]
            + summed.shape[(-(len(self._param_shape) + batch_ndims)) :],
        )
        interaction_indices = tf.transpose(tf.convert_to_tensor(interaction_indices))
        indices = tf_ravel_multi_index(interaction_indices, self._interaction_shape)
        summed = tf.gather_nd(summed, indices[:, tf.newaxis])

        rank = len(summed.shape.as_list())
        # move parameter batch dims back to the front
        permutation = list(range(rank - batch_ndims, rank)) + list(
            range(rank - batch_ndims)
        )
        summed = tf.transpose(summed, permutation)

        return summed

    def shape(self):
        return self._intrinsic_shape

    def retrieve_indices(self, data):
        return self._interactions.retrieve_indices(data)


class MultiwayContingencyTable(object):
    def __init__(self, interaction) -> None:
        self.interaction = interaction

    def fit(self, data_factory, dtype=tf.int32):
        decomposition = Decomposed(self.interaction, [1])
        n_dims = np.prod(decomposition._interaction_shape)
        counts = tf.zeros((n_dims), dtype=dtype)
        dataset = data_factory()
        counter = CountEncoder(list(range(np.prod(decomposition._interaction_shape))))
        for batch in tqdm(iter(dataset)):
            indices = decomposition.retrieve_indices(batch)
            if isinstance(indices, tf.RaggedTensor):
                indices = indices.to_tensor()
            indices = tf_ravel_multi_index(
                tf.transpose(indices), decomposition._interaction_shape
            )
            counts_ = counter.encode(indices)
            counts += tf.cast(counts_[1:], counts.dtype)
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
        otheraxes = [d for d in range(len(self.interaction._dimensions)) if d not in axes]
        counts = np.reshape(self.counts, self.interaction._intrinsic_shape)
        counts = np.apply_over_axes(
            np.sum,
            counts,
            [
                d
                for d in range(len(self.interaction._intrinsic_shape))
                if d not in axes
            ],
        )
        counts = np.transpose(counts, axes + otheraxes)
        counts = np.reshape(counts, counts.shape[:len(axes)])
        return counts


def demo():
    dims = [
        ("planned", ["no", "yes"]),
        ("pre2011", 2),
        ("mdc", 26),
        *[(f"hx_{j}", ["low", "high"]) for j in range(5)],
    ]

    interact = Interactions(dims, exclusions=[]).truncate_to_order(3)
    p = Decomposed(
        interactions=interact, param_shape=[100, 1], name="beta", implicit=True
    )
    print(interact)
    print(p.generate_labels()["nono"])
    t, n, s = p.generate_tensors(batch_shape=[4])
    out = p.sum_parts(t)

    r = p.lookup(indices, t)
    r1 = p._lookup_by_parts(indices, t)
    out2 = p.dot(indices)

    interact = Interactions([], exclusions=[])
    p0 = Decomposed(
        interactions=interact, param_shape=[100], name="beta", implicit=True
    )

    print(p0)

    indices = [
        [0, 0, 21, 1, 1, 1, 1, 0],
        [0, 0, 12, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 13, 1, 0, 1, 1, 0],
        [0, 0, 13, 0, 0, 1, 1, 1],
        [0, 1, 13, 0, 0, 1, 0, 1],
    ]

    return None


if __name__ == "__main__":
    demo()
