#!/usr/bin/env python3
from itertools import product

import jax.numpy as jnp
import numpy as np
from tensorflow_probability.python.internal.backend.numpy import gather_nd


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


if __name__ == "__main__":
    main()
