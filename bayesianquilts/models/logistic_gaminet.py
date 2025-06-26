import numpy as np
import tensorflow_probability as tfp
from tensorflow_probability.python import bijectors as tfb
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.model import BayesianModel
from bayesianquilts.models.logistic_relunet import LogisticRelunet


class GamiNetUnivarite(LogisticRelunet):
    def __init__(self, **kwargs):
        pass

    def predictive_distribution(self, data, **params):

        X = tf.cast(
            data["X"],
            self.dtype,
        )

        logits = self.eval(X, params)
        logits = tf.pad(logits, [(0, 0)]*(len(logits.shape) -1) + [(1, 0)])
        rv_outcome = tfd.Categorical(logits=logits)
        log_likelihood = rv_outcome.log_prob(tf.squeeze(data[self.outcome_label]))

        return {
            "log_likelihood": log_likelihood,
            "prediction": rv_outcome,
            "logits": logits,
        }

class GamiNetPairwise(LogisticRelunet):
    pass
