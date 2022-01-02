#!/usr/bin/env python3
import re
from collections import Counter, defaultdict
from itertools import product
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.check_ops import assert_none_equal_v2
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

from bayesianquilts.stackedtensor import broadcast_tensors


def tf_ravel_multi_index(multi_index, dims):
    strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
    multi_index = tf.cast(multi_index, strides.dtype)
    return tf.reduce_sum(multi_index * tf.expand_dims(strides, 1), axis=0)


class Interactions(object):
    _interaction_list = []

    def __init__(self, dimensions, exclusions=None) -> None:
        super().__init__()
        self._dimensions = dimensions
        self._intrinsic_shape = [x[1] for x in self._dimensions]
        self._exclusions = [set(s) for s in exclusions]

    def shape(self):
        return self._intrinsic_shape

    def rank(self):
        return len(self.shape())

    def __print__(self):
        print(f"shape: {self._intrinsic_shape}")
        print(f"dimensions: {self._dimensions}")
        print(f"exclusions: {self._exclusions}")

    def __str__(self):
        out = f"Interaction dimenions: {self._dimensions}"
        return out


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
        param_shape=[],
        default_val=None,
        dtype=tf.float32,
        name="",
        *args,
        **kwargs,
    ) -> None:

        super().__init__()
        assert isinstance(
            interactions, Interactions
        ), "Instantiation requires a parameterization"

        self._interactions = interactions
        self._default_val = default_val
        self._interaction_shape = interactions.shape()
        self._intrinsic_shape = self._interaction_shape + param_shape
        self._dtype = dtype
        self._name = name
        self._param_shape = param_shape
        (
            self._tensor_parts,
            self._tensor_part_interactions,
            self._tensor_part_shapes,
        ) = self.generate_tensors()

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
            value = tf.reshape(
                value,
                batch_shape
                + [np.prod(t_shape[: (-len(self._param_shape))])]
                + self._param_shape,
            )
            tensors[self._name + "__" + interaction_name] = value

        return tensors, tensor_names, tensor_shapes

    def set_params(self, tensors):
        for k in self._param_tensors.keys():
            self._param_tensors[k] = tensors[k]

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

    def sum_parts(self, tensors=None, unravel=False):
        tensors = self._tensor_parts if tensors is None else tensors
        partial_sum = tf.zeros(self.shape(), self._dtype)
        #  batch

        # folded
        for t in tf.nest.flatten(tensors):
            partial_sum = tf.add_n(
                broadcast_tensors([partial_sum, tf.cast(t, self._dtype)])
            )

        return partial_sum

    def unravel_tensor(self, tensor):
        pass

    def __str__(self):
        out = f"Parameter shape: {self._param_shape} \n"
        out += f"{self._interactions} \n"
        out += f"Component tensors: {len(self._param_tensors.keys())} \n"
        out += f"Effective parameter cardinality: {np.prod(self._intrinsic_shape)} \n"
        out += f"Actual parameter cardinality: {sum([np.prod(t) for t in self._tensor_part_shapes.values()])}\n"
        return out

    def tensor_keys(self):
        return sorted(list(self._param_tensors.keys()))

    def lookup(self, interaction_indices, tensors=None):
        # flatten the indices
        interaction_indices = tf.convert_to_tensor(interaction_indices)
        # assert interaction_indices.shape.as_list()[-1] == len(self._interaction_shape)

        interaction_shape = tf.convert_to_tensor(
            self._interaction_shape, dtype=interaction_indices.dtype
        )

        summed = self.sum_parts(tensors, ravel=True)

        overall_shape = summed.shape.as_list()
        rank = len(overall_shape)
        batch_ndims = rank - len(self.shape())

        summed = tf.reshape(
            summed,
            overall_shape[:batch_ndims]
            + [np.prod(self._interaction_shape)]
            + self._param_shape,
        )

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


def main():
    dims = [
        ("planned", 2),
        ("pre2011", 2),
        ("mdc", 26),
        *[(f"hx_{j}", 2) for j in range(5)],
    ]
    onehot = list(filter(lambda x: sum(x) > 4, product([0, 1], repeat=len(dims))))
    exclusions = [*[(f"hx_{j}",) for j in range(5)]]
    hot = [set(d[0] for i, d in zip(ind, dims) if i == 1) for ind in onehot]
    exclusions += hot
    # get rid of anything higher than 3rd order

    interact = Interactions(dims, exclusions=exclusions)
    print(interact)
    p = Decomposed(interactions=interact, param_shape=[100], name="beta")
    print(p)
    indices = [
        [0, 0, 21, 1, 1, 1, 1, 0],
        [0, 0, 12, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 13, 1, 0, 1, 1, 0],
        [0, 0, 13, 0, 0, 1, 1, 1],
    ]

    t, n, s = p.generate_tensors(batch_shape=[4])
    r = p.lookup(indices, t1)


if __name__ == "__main__":
    main()
