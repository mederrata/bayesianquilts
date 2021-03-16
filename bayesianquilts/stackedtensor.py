#!/usr/bin/env python3

import six
import numpy as np

import tensorflow as tf


def broadcast_tensors(tensors):
    shapes = [t.get_shape().as_list() for t in tensors]
    max_rank = max([len(s) for s in shapes])
    # Rank equalize all the tensors
    for index in range(len(shapes)):
        shape = shapes[index]
        if len(shape) == max_rank:
            continue

        tensor = tensors[index]
        for _ in range(max_rank - len(shape)):
            shape.insert(0, 1)
            tensor = tf.expand_dims(tensor, axis=0)
        tensors[index] = tensor

    # Ensure if broadcasting is possible
    from collections import Counter
    broadcast_shape = []
    for index in range(max_rank):
        dimensions = [s[index] for s in shapes]
        repeats = Counter(dimensions)
        if len(repeats) > 2 or (len(repeats) == 2 and
                                1 not in list(repeats.keys())):
            raise Exception("Broadcasting not possible")
        broadcast_shape.append(max(repeats.keys()))

    # Broadcast the tensors
    for axis, dimension in enumerate(broadcast_shape):
        tensors = [tf.concat([t] * dimension, axis=axis)
                   if t.get_shape()[axis] == 1 else t for t in tensors]

    return tensors
