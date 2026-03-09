import functools
import inspect
import typing

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from tensorflow_probability.substrates.jax import monte_carlo
from tensorflow_probability.substrates.jax.vi import (GradientEstimators,
                                                      csiszar_divergence,
                                                      kl_reverse)

from bayesianquilts.util import training_loop


def minibatch_mc_variational_loss(
    target_log_prob_fn,
    surrogate_posterior,
    dataset_size,
    batch_size,
    data,
    sample_size=1,
    sample_batches=1,
    importance_sample_size=1,
    discrepancy_fn=kl_reverse,
    use_reparameterization=None,
    gradient_estimator=None,
    stopped_surrogate_posterior=None,
    seed=None,
    cost="reweighted",
    stl=True,
    **kwargs,
):
    """Minibatch variational loss (per-datum negative ELBO).

    Computes the standard minibatch ELBO normalized by dataset size:

        loss = (1/N) * E_q[log q(theta) - log p(theta) - (N/B) * log p(y_batch | theta)]

    where N is the total dataset size and B is the minibatch size.

    Dividing by N makes the loss and its gradients O(1) regardless of dataset
    size, so a single learning rate works across datasets of different sizes.

    The Sticking-the-Landing (STL) estimator (Roeder et al., 2017) is used
    by default, which applies stop_gradient to log q(theta) to reduce
    gradient variance without bias.

    Args:
        target_log_prob_fn: Function(data, prior_weight, **params) -> scalar.
            Should return log p(y_batch | theta) + prior_weight * log p(theta).
        surrogate_posterior: TFP distribution representing q(theta).
        dataset_size: Total number of data points N.
        batch_size: Minibatch size B.
        data: Current minibatch of data.
        sample_size: Number of Monte Carlo samples from q per ELBO estimate.
        sample_batches: Number of independent sample batches to average over.
        seed: JAX PRNG key for reproducible sampling. If None, falls back to
            numpy-based key generation (not reproducible under JIT).
        stl: If True (default), use the STL gradient estimator which applies
            stop_gradient to log q(theta). This reduces gradient variance
            with zero computational overhead.
        cost: "reweighted" (default) or "tfp" for TFP's importance-weighted
            divergence.

    Returns:
        Scalar per-datum loss value (negative ELBO / N).
    """
    scale = dataset_size / batch_size

    def sample_elbo(key):
        q_samples, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
            sample_size, seed=key
        )
        return q_samples, q_lp

    batch_expectations = []

    reweighted = functools.partial(
        target_log_prob_fn,
        data=data,
        prior_weight=jnp.astype(batch_size / dataset_size, jnp.float64),
    )

    def sample_expected_elbo(q_samples, q_lp):
        # target_log_prob_fn returns: log_lik(batch) + (B/N) * log_prior
        # Multiply by N/B:           (N/B)*log_lik(batch) + log_prior
        # Loss = E_q[log q - (N/B)*log_lik - log_prior]
        #      = E_q[log q - scale * target_log_prob]
        penalized_ll = target_log_prob_fn(
            data=data,
            prior_weight=batch_size / dataset_size,
            **q_samples,
        )
        # STL: stop gradient through log q(theta) to reduce variance
        q_lp_grad = jax.lax.stop_gradient(q_lp) if stl else q_lp
        expected_elbo = jnp.mean(q_lp_grad - scale * penalized_ll)
        return expected_elbo

    for i in range(sample_batches):
        if seed is not None:
            batch_key = random.fold_in(seed, i)
        else:
            batch_key = random.PRNGKey(np.random.randint(0, 2**31))
        q_samples, q_lp = sample_elbo(batch_key)
        if cost == "tfp":
            batch_expectations += [
                monte_carlo.expectation(
                    f=csiszar_divergence._make_importance_weighted_divergence_fn(
                        reweighted,
                        surrogate_posterior=surrogate_posterior,
                        discrepancy_fn=discrepancy_fn,
                        precomputed_surrogate_log_prob=q_lp * batch_size / dataset_size,
                        importance_sample_size=importance_sample_size,
                        gradient_estimator=gradient_estimator,
                        stopped_surrogate_posterior=(stopped_surrogate_posterior),
                    ),
                    samples=q_samples,
                    log_prob=surrogate_posterior.log_prob,
                    use_reparameterization=(
                        gradient_estimator != GradientEstimators.SCORE_FUNCTION
                    ),
                )
            ]
        else:
            batch_expectations += [sample_expected_elbo(q_samples, q_lp)]
            batch_expectations = jnp.atleast_1d(batch_expectations)
    batch_expectations = jnp.mean(batch_expectations, axis=0)
    return batch_expectations / dataset_size


def minibatch_fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_generator,
    surrogate_initializer,
    data_iterator,
    dataset_size: int,
    initial_values: typing.Dict = None,
    batch_size: int = 1,
    steps_per_epoch: int = 1,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
    check_convergence_every: int = 1,
    sample_size=8,
    sample_batches=1,
    lr_decay_factor: float = 0.5,
    learning_rate=1.0,
    patience: int = 3,
    name=None,
    test_fn=None,
    clip_norm: float = None,
    zero_nan_grads: bool = False,
    snapshot_epoch: int | None = None,
    seed: int | None = None,
    stl: bool = True,
    **kwargs,
):
    if initial_values is None:
        initial_values = surrogate_initializer()

    # Set up PRNG key for reproducible sampling
    if seed is not None:
        base_key = random.PRNGKey(seed)
    else:
        base_key = None

    def complete_variational_loss_fn(data=None, params=None, seed=None):
        """Loss function called in the optimization loop on each minibatch."""
        if params is None:
            params = initial_values

        return minibatch_mc_variational_loss(
            target_log_prob_fn,
            surrogate_generator(params),
            sample_size=sample_size,
            sample_batches=sample_batches,
            data=data,
            dataset_size=dataset_size,
            batch_size=batch_size,
            seed=seed,
            stl=stl,
            name=name,
            **kwargs,
        )

    return training_loop(
        loss_fn=complete_variational_loss_fn,
        base_optimizer_fn=None,
        initial_values=initial_values,
        data_iterator=data_iterator,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        accumulation_steps=accumulation_steps,
        check_convergence_every=check_convergence_every,
        learning_rate=learning_rate,
        patience=patience,
        lr_decay_factor=lr_decay_factor,
        clip_norm=clip_norm,
        zero_nan_grads=zero_nan_grads,
        snapshot_epoch=snapshot_epoch,
        seed=seed,
        **kwargs,
    )
