from abc import ABC, abstractclassmethod

import numpy as np
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts import BayesianModel
from bayesianquilts.tf.parameter import Decomposed


class Ensemble(ABC):
    weights = []

    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models))/len(models)
        else:
            self.weights = np.array(weights)

    @abstractclassmethod
    def fit(self, data_factory, metric):
        pass
    
    def fit_children(self, *args, **kwargs):
        pass

class ModelSelector(Ensemble):
    pass


class ParametricEnsemble(Ensemble):
    pass
