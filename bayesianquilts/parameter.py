#!/usr/bin/env python3
import re
from collections import Counter, defaultdict
from itertools import product, chain, combinations
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd


class InteractionParameterization(object):
    _interaction_list = []
    def __init__(
            self,
            dimensions) -> None:
        super().__init__()
        self._dimensions = dimensions
        self._intrinsic_shape = [
            x[1] for x in self._dimensions
        ]

    def shape(self):
        return self._intrinsic_shape
    
    def exclude(self, interaction):
        pass


class DecomposedParam(object):
    _param_tensors = {}
    _intrinsic_shape = None

    def __init__(
            self,
            interactions,
            default_val=None,
            dtype=tf.float32,
            *args,
            **kwargs) -> None:
        super().__init__()
        assert isinstance(
            interactions, InteractionParameterization
            ), "Instantiation requires a parameterization"
        
        self._interactions = interactions
        self._default_val = default_val
        self._dtype = dtype

    def set_params(self, tensors):
        self._param_tensors = tensors

    def __add__(self, x):
        return self.constitute().__add__(x)

    def __radd__(self, x):
        return x + self.constitute()

    def __mul__(self, x):
        return self.constitute().__mul__(x)

    def constitute(self):
        partial_sum = tf.zeros(
            self.shape(),
            self._dtype)
        for t in tf.nest.flatten(self._param_tensors):
            partial_sum += t
        return partial_sum

    def tensor_keys(self):
        return sorted(list(self._param_tensors.keys()))

    def query(self, indices):
        pass
    
    def shape(self):
        return self._intrinsic_shape


def main():
    interact = InteractionParameterization(
        [("Dx", 100), ("Tx", 32), ("Hx", 3)]
    )
    pass


if __name__ == "__main__":
    main()
