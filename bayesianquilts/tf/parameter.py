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
        self._param_tensors, self._param_interactions = self.generate_tensors()
        self._param_shapes = {k: v.shape for k, v in self._param_tensors.items()}

    def generate_tensors(
        self, batch_shape=None, target=None, flatten_indices=False, dtype=None
    ):
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
            tensors[self._name + "__" + interaction_name] = value
            residual = tf.add_n(broadcast_tensors([residual, -1.0 * value]))

        if flatten_indices:
            tensors = self.flatten_indices(tensors)

        return tensors, tensor_names

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
                self._param_shapes[k]
            except KeyError:
                continue
            rank = len(tensors[k].shape.as_list())
            batch_shape = tensors[k].shape.as_list()[: (rank - param_rank - 1)]
            tensors[k] = tf.reshape(tensors[k], batch_shape + self._param_shapes[k])
        return tensors

    def __add__(self, x):
        if tf.is_tensor(x):
            return tf.add_n(broadcast_tensors([x, self.constitute()]))
        return self.constitute().__add__(x)

    def __radd__(self, x):
        return self.__add__(x)

    def __mul__(self, x):
        return self.constitute().__mul__(x)

    def shape(self):
        return self._intrinsic_shape

    def constitute(self, tensors=None):
        tensors = self._param_tensors if tensors is None else tensors
        partial_sum = tf.zeros(self.shape(), self._dtype)
        #  batch

        # folded
        for t in tf.nest.flatten(self._param_tensors):
            partial_sum = tf.add_n(
                broadcast_tensors([partial_sum, tf.cast(t, self._dtype)])
            )
        return partial_sum
    
    def __str__(self):
        out = f"Parameter shape: {self._param_shape} \n" 
        out += f"{self._interactions} \n"
        out += f"Component tensors: {len(self._param_tensors.keys())} \n"
        out += f"Effective parameter cardinality: {np.prod(self._intrinsic_shape)} \n"
        out += f"Actual parameter cardinality: {sum([np.prod(t.shape.as_list()) for t in self._param_shapes.values()])}\n"
        return out

    def tensor_keys(self):
        return sorted(list(self._param_tensors.keys()))

    def query(self, interaction_indices, tensors=None):
        _global = self.constitute(tensors)
        rank = len(_global.shape.as_list())

        batch_ndims = rank - len(self.shape())

        # stick batch axes on the  end
        permutation = list(range(batch_ndims, rank)) + list(range(batch_ndims))
        _global = tf.transpose(_global, permutation)
        _global = tf.reshape(
            _global,
            [np.prod(_global.shape[: -(len(self._param_shape) + batch_ndims)])]
            + _global.shape[(-(len(self._param_shape) + batch_ndims)) :],
        )
        interaction_indices = tf.transpose(tf.convert_to_tensor(interaction_indices))
        indices = tf_ravel_multi_index(interaction_indices, self._interaction_shape)
        _global = tf.gather_nd(_global, indices[:, tf.newaxis])
        
        rank = len(_global.shape.as_list())
        # move parameter batch dims back to the front
        permutation = list(range(rank - batch_ndims, rank)) + list(
            range(rank - batch_ndims)
        )
        _global = tf.transpose(_global, permutation)

        return _global

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
    p = Decomposed(interactions=interact, param_shape=[20, 2], name="beta")
    print(p)
    indices = [
        [0, 0, 21, 1, 1, 1, 1, 0],
        [0, 0, 12, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 13, 1, 0, 1, 1, 0],
        [0, 0, 13, 0, 0, 1, 1, 1],
    ]

    t, n = p.generate_tensors(batch_shape=[4], flatten_indices=True)
    t1 = p.inflate_indices(t)
    p.set_params(t1)
    r = p.query(indices, t)
    
    @tf.function
    def graph_test(tensors):
        return p.query(indices, tensors)
    
    test = graph_test(t)


if __name__ == "__main__":
    main()
