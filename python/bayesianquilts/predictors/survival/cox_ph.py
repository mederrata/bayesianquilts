#!/usr/bin/env python3
"""Bayesian Cox Proportional Hazards model for the bayesianquilts framework."""

import jax
import jax.numpy as jnp
import jax.flatten_util
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin
from bayesianquilts.model import BayesianModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator


def _cox_partial_log_likelihood(beta, X, time, event, dtype=jnp.float64):
    """Compute Cox partial log-likelihood.

    Uses the Breslow method: sort descending by time, then cumulative logsumexp.

    Parameters
    ----------
    beta : array, shape (..., D)
        Regression coefficients. Leading dims are sample dims.
    X : array, shape (N, D)
        Covariate matrix.
    time : array, shape (N,)
        Observed times.
    event : array, shape (N,)
        Event indicators (1 = event, 0 = censored).

    Returns
    -------
    ll : array, shape (..., N)
        Per-observation log-likelihood contribution (0 for censored).
    """
    X = jnp.asarray(X, dtype)
    time = jnp.asarray(jnp.squeeze(time), dtype)
    event = jnp.asarray(jnp.squeeze(event), dtype)

    # Sort by time descending so cumulative logsumexp builds risk sets
    order = jnp.argsort(-time)
    X_sorted = X[order]
    event_sorted = event[order]

    # eta = X @ beta^T -> shape (..., N)
    # beta: (..., D), X_sorted: (N, D)
    eta_sorted = jnp.einsum("...d,nd->...n", beta, X_sorted)

    # Cumulative logsumexp along the sorted axis (axis=-1, left to right)
    # For descending-sorted times, cum_logsumexp[i] = logsumexp(eta[0:i+1])
    # which corresponds to the risk set R(t_i) = {j : t_j >= t_i}
    cum_lse = _cumulative_logsumexp(eta_sorted, axis=-1)

    # Per-observation contribution: event * (eta - cum_logsumexp)
    ll_sorted = event_sorted * (eta_sorted - cum_lse)

    # Unsort back to original order
    inv_order = jnp.argsort(order)
    ll = ll_sorted[..., inv_order]

    return ll


def _cumulative_logsumexp(x, axis=-1):
    """Numerically stable cumulative logsumexp along an axis.

    Parameters
    ----------
    x : array
        Input array.
    axis : int
        Axis along which to accumulate.

    Returns
    -------
    result : array, same shape as x
        result[..., i] = logsumexp(x[..., :i+1])
    """
    # Use associative scan for efficiency
    # logsumexp(a, b) = max(a, b) + log(exp(a - max(a,b)) + exp(b - max(a,b)))
    def _binary_logsumexp(a, b):
        return jnp.logaddexp(a, b)

    return jax.lax.associative_scan(_binary_logsumexp, x, axis=axis)


class CoxPH(BayesianModel):
    """Bayesian Cox Proportional Hazards model.

    Implements the Cox partial likelihood with a Normal prior on
    the regression coefficients beta.

    Parameters
    ----------
    dim_regressors : int
        Number of covariates.
    prior_scale : float
        Scale of the Normal prior on beta.
    global_rank : int
        Rank for low-rank surrogate posterior approximation.
        0 means fully factored (mean-field).
    dtype : jnp.dtype
        Data type for computations.
    """

    distribution = None
    surrogate_distribution = None
    reparameterized = True

    def __init__(
        self,
        dim_regressors,
        prior_scale=1.0,
        global_rank=0,
        dtype=jnp.float64,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.dim_regressors = dim_regressors
        self.prior_scale = prior_scale
        self._global_rank = global_rank

        self.create_distributions()

    def create_distributions(self):
        """Build prior, bijectors, and surrogate posterior generator."""
        distribution_dict = {}
        bijectors = {}

        distribution_dict["beta"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros(self.dim_regressors, dtype=self.dtype),
                scale=self.prior_scale
                * jnp.ones(self.dim_regressors, dtype=self.dtype),
            ),
            reinterpreted_batch_ndims=1,
        )
        bijectors["beta"] = tfb.Identity()

        self.bijectors = bijectors
        self.prior_distribution = tfd.JointDistributionNamed(distribution_dict)
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.prior_distribution,
                bijectors,
                dtype=self.dtype,
                global_rank=self._global_rank,
            )
        )
        self.params = self.surrogate_parameter_initializer()
        self.var_list = list(self.prior_distribution.model.keys())

    def predictive_distribution(self, data, **params):
        """Compute per-observation log-likelihood and linear predictor.

        Parameters
        ----------
        data : dict
            Must contain 'X' (N, D), 'time' (N,), 'event' (N,).
        **params : dict
            Must contain 'beta' with shape (..., D).

        Returns
        -------
        dict with 'log_likelihood' and 'prediction' (linear predictor).
        """
        beta = jnp.asarray(params["beta"], self.dtype)
        X = jnp.asarray(data["X"], self.dtype)
        time = data["time"]
        event = data["event"]

        ll = _cox_partial_log_likelihood(beta, X, time, event, self.dtype)
        eta = jnp.einsum("...d,nd->...n", beta, X)

        return {
            "log_likelihood": ll,
            "prediction": eta,
        }

    def log_likelihood(self, data, **params):
        """Return per-observation log-likelihood, shape (..., N)."""
        return self.predictive_distribution(data, **params)["log_likelihood"]

    def unormalized_log_prob(self, data=None, prior_weight=1.0, **params):
        """Unnormalized log posterior: sum of log-lik + prior."""
        log_lik = self.log_likelihood(data, **params)

        if log_lik.ndim > 1:
            total_ll = jnp.sum(log_lik, axis=-1)
        else:
            total_ll = jnp.sum(log_lik)

        prior = self.prior_distribution.log_prob(params)
        return total_ll + prior_weight * prior

    def fit(
        self,
        batched_data_factory,
        batch_size=None,
        dataset_size=None,
        num_epochs=100,
        learning_rate=0.005,
        sample_size=8,
        **kwargs,
    ):
        """Fit the model using minibatch ADVI.

        Delegates to the base-class ``_calibrate_minibatch_advi`` via
        ``BayesianModel.fit``, which builds an infinite data iterator
        from *batched_data_factory* and calls
        ``minibatch_fit_surrogate_posterior``.

        Parameters
        ----------
        batched_data_factory : callable
            Zero-argument callable that returns an iterator of data batches.
        batch_size : int
            Number of observations per mini-batch.
        dataset_size : int
            Total number of observations in the training set.
        num_epochs : int
            Number of training epochs.
        learning_rate : float
            Initial learning rate.
        sample_size : int
            Number of Monte-Carlo samples for ELBO estimation.
        **kwargs
            Forwarded to ``_calibrate_minibatch_advi``.
        """
        return super().fit(
            batched_data_factory,
            batch_size=batch_size,
            dataset_size=dataset_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            sample_size=sample_size,
            **kwargs,
        )


class CoxPHLikelihood(AutoDiffLikelihoodMixin):
    """AIS-compatible likelihood wrapper for :class:`CoxPH`.

    Implements ``log_likelihood``, ``extract_parameters``, and
    ``reconstruct_parameters`` so that
    :class:`~bayesianquilts.metrics.ais.AdaptiveImportanceSampler` can
    compute LOO-CV diagnostics for a fitted Cox model.
    """

    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype

    def log_likelihood(self, data, params):
        """Per-observation log-likelihood.

        Parameters
        ----------
        data : dict
            Must contain 'X', 'time', 'event'.
        params : dict
            Must contain 'beta' with shape (S, D).

        Returns
        -------
        array, shape (S, N)
        """
        return self.model.log_likelihood(data, **params)

    def extract_parameters(self, params):
        """Flatten parameter dict to array of shape (S, D).

        For CoxPH, the only parameter is ``beta`` which is already (S, D),
        but we go through the generic ravel path to stay consistent with
        the rest of the framework.
        """
        flat_params = jax.vmap(
            lambda p: jax.flatten_util.ravel_pytree(p)[0]
        )(params)
        return flat_params

    def reconstruct_parameters(self, flat_params, template):
        """Unflatten array back to parameter dict.

        Parameters
        ----------
        flat_params : array, shape (..., K)
            Flattened parameters.
        template : dict
            Template parameter dict (single sample) for structure recovery.

        Returns
        -------
        dict matching the structure of *template*.
        """
        # Strip leading sample dim from template if present
        if isinstance(template.get("beta"), jnp.ndarray) and template["beta"].ndim > 1:
            template = jax.tree_util.tree_map(lambda x: x[0], template)

        dummy_flat, unflatten = jax.flatten_util.ravel_pytree(template)
        K = dummy_flat.shape[0]
        input_shape = flat_params.shape

        if input_shape[-1] != K:
            raise ValueError(
                f"Last dimension {input_shape[-1]} != expected K={K}"
            )

        batch_dims = input_shape[:-1]
        n_batch = 1
        for d in batch_dims:
            n_batch *= d

        flat_reshaped = flat_params.reshape((n_batch, K))
        unflattened_flat = jax.vmap(unflatten)(flat_reshaped)

        def reshape_leaf(leaf):
            leaf_param_shape = leaf.shape[1:]
            return leaf.reshape(batch_dims + leaf_param_shape)

        return jax.tree_util.tree_map(reshape_leaf, unflattened_flat)
