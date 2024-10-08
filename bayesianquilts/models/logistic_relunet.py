#!/usr/bin/env python3


import tensorflow as tf
import tensorflow_probability as tfp

from bayesianquilts.nn.dense import DenseHorseshoe

tfd = tfp.distributions


class LogisticRelunet(DenseHorseshoe):
    def __init__(
        self,
        dim_regressors: int,
        layer_sizes: list[int] | None = None,
        regressor_scales: list[float] | None = None,
        regressor_offsets: list[float] | None = None,
        outcome_classes: int = 2,
        dtype: tf.DType = tf.float64,
        outcome_label: str = "y",
        **kwargs
    ):
        if layer_sizes is None:
            layer_sizes = ([int(dim_regressors / 10), 20, outcome_classes - 1],)
        else:
            layer_sizes += [outcome_classes - 1]
        super(LogisticRelunet, self).__init__(
            input_size=dim_regressors,
            layer_sizes=layer_sizes,
            activation_fn=tf.nn.relu,
            weight_scale=1.0,
            bias_scale=1.0,
            dtype=dtype,
            **kwargs
        )
        if regressor_scales is None:
            self.regressor_scales = 1
        else:
            self.regressor_scales = regressor_scales
        self.regressor_offsets = (
            regressor_offsets if regressor_offsets is not None else 0
        )
        self.outcome_classes = outcome_classes
        self.outcome_label = outcome_label

    def predictive_distribution(self, data: dict[str, tf.Tensor], **params):

        X = tf.cast(
            data["X"],
            self.dtype,
        )

        logits = self.eval(X, params)
        logits = tf.pad(logits, [(0, 0)] * (len(logits.shape) - 1) + [(1, 0)])
        rv_outcome = tfd.Categorical(logits=logits)
        log_likelihood = rv_outcome.log_prob(tf.squeeze(data[self.outcome_label]))

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": logits,
        }

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(
        self,
        data: dict[str, tf.Tensor] = None,
        prior_weight: tf.Tensor | float = tf.constant(1.0),
        **params
    ):
        log_likelihood = self.log_likelihood(data, **params)
        finite_portion = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.zeros_like(log_likelihood),
        )
        min_val = tf.reduce_min(finite_portion) - 1.0
        max_val = 0.0
        log_likelihood = tf.clip_by_value(log_likelihood, min_val, max_val)
        log_likelihood = tf.where(
            tf.math.is_finite(log_likelihood),
            log_likelihood,
            tf.ones_like(log_likelihood) * min_val,
        )
        prior = self.prior_distribution.log_prob(params)
        return (
            tf.reduce_sum(log_likelihood, axis=-1)
            + tf.cast(prior_weight, prior.dtype) * prior
        )

    def stepaway_loo(
        self, data: dict[str, tf.Tensor], params: dict[str, tf.Tensor], step_size: float
    ):
        """Compute step-away transformation for LOO

        Keyword arguments:
        argument -- description
        Return: return_description
        """

        y = tf.cast(tf.squeeze(data["y"]), tf.float64)
        X = tf.cast(data["X"], tf.float64)
        mu = self.eval(X, params)[..., 0]
        sigma = tf.math.sigmoid(mu)
        nu = (y * (sigma) + (1 - y) * (1 - sigma)) ** -1
        nu_weights = nu / tf.reduce_sum(nu, axis=0, keepdims=True)
        p_loo = tf.reduce_sum(sigma * nu_weights, axis=0)
        ll_loo = tf.reduce_sum(nu_weights / nu, axis=0)

        z1 = params["w_0"][..., tf.newaxis, :, :] * X[..., tf.newaxis]
        z1 = tf.reduce_sum(z1, axis=-2)
        az1 = tf.nn.relu(z1)

        grad_pre = tf.cast(y, tf.float64) * tf.cast(1 - sigma, tf.float64) - tf.cast(
            1 - y, tf.float64
        ) * tf.cast(sigma, tf.float64)

        return {
            "p_loo": p_loo,
            "p": tf.reduce_mean(sigma, axis=0),
            "nu_weights": nu_weights,
            "ll_loo": ll_loo,
        }
