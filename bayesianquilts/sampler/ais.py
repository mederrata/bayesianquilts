#!/usr/bin/env python3

import abc
import typing
from abc import ABC

import tensorflow as tf


class LikelihoodModel(ABC):
    @abc.abstractmethod
    def log_like(
        self, data: typing.Dict[str, tf.Tensor], params: typing.Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        pass


class StepTransformation(ABC):
    def __init__(self, hbar):
        self.hbar = hbar

    @abc.abstractmethod
    def step(
        self, params: typing.Dict[str, tf.Tensor]
    ) -> typing.Tuple[typing.Dict[str, tf.Tensor], tf.Tensor]:  # step, jacobian
        return {}


class AntiLogLikelihoodStep(StepTransformation):

    def step(
        self, params: typing.Dict[str, tf.Tensor]
    ) -> typing.Tuple[typing.Dict[str, tf.Tensor], tf.Tensor]:  # step, jacobian
        return {}


class KlDescentStep(StepTransformation):
    def step(
        self, params: typing.Dict[str, tf.Tensor]
    ) -> typing.Tuple[typing.Dict[str, tf.Tensor], tf.Tensor]:  # step, jacobian
        return {}


class VarianceDescentStep(StepTransformation):
    def step(
        self, params: typing.Dict[str, tf.Tensor]
    ) -> typing.Tuple[typing.Dict[str, tf.Tensor], tf.Tensor]:  # step, jacobian
        return {}


class SigmoidalModel(LikelihoodModel):

    def log_like(
        self, data: typing.Dict[str, tf.Tensor], params: typing.Dict[str, tf.Tensor]
    ) -> tf.Tensor:
        mu, _ = self.mu(data, params)
        sigma = tf.math.sigmoid(mu)
        log_ell = tf.math.xlogy(data["y"], sigma) + tf.math.xlogy(
            1 - data["y"], 1 - sigma
        )  # S x N

        return log_ell

    def grad_log_like(
        self, data: typing.Dict[str, tf.Tensor], params: typing.Dict[str, tf.Tensor]
    ) -> typing.Dict[str, tf.Tensor]:
        mu, grad_mu = self.mu(data, params)
        sigma = tf.math.sigmoid(mu)
        prefactor = data["y"] * (1 - sigma) - sigma * (1 - data["y"])  # S x N

        grad_ll = {k: gmu * prefactor for k, gmu in grad_mu.items()}

        return grad_ll

    @abc.abstractmethod
    def mu(
        self, data: typing.Dict[str, tf.Tensor], params: typing.Dict[str, tf.Tensor]
    ) -> typing.Tuple[tf.Tensor, typing.Dict[str, tf.Tensor]]:
        return


class LogisticRegression(SigmoidalModel):
    def mu(
        self, data: typing.Dict[str, tf.Tensor], params: typing.Dict[str, tf.Tensor]
    ) -> typing.Tuple[tf.Tensor, typing.Dict[str, tf.Tensor]]:
        m = params["alpha"] + tf.reduce_sum(
            params["beta"][..., tf.newaxis] * data["X"], axis=-1
        )
        grad = {}
        grad["alpha"] = tf.ones_like(tf.reduce_sum(data["X"], axis=-1))
        grad["beta"] = data["X"]
        return m, grad


class ReluLogisticNn(SigmoidalModel):
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes


class AdaptiveIsSampler(ABC):
    def __init__(self, likelihood, transformation, grad_likelihood=None):
        self.likelihood = likelihood
        self.grad_likelihood = grad_likelihood
        self.transformation = transformation

    def adapt(self, params, model: LikelihoodModel, data: typing.Dict[str, tf.Tensor]):
        return params
