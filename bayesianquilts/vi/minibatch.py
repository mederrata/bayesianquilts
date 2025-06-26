import functools
import typing

import jax.numpy as jnp
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
    **kwargs,
):
    """The minibatch variational loss

    Args:
        target_log_prob_fn (_type_): log_likelihood + prior_weight*log_prior
        surrogate_posterior (_type_): _description_
        dataset_size (_type_): _description_
        batch_size (_type_): _description_
        sample_size (int, optional): _description_. Defaults to 1.
        discrepancy_fn (_type_, optional): _description_. Defaults to tfp.vi.kl_reverse.
        seed (_type_, optional): _description_. Defaults to None.
        data (_type_, optional): _description_. Defaults to None.
        name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    def sample_elbo():
        q_samples, q_lp = surrogate_posterior.experimental_sample_and_log_prob(
            sample_size, seed=seed
        )
        return q_samples, q_lp

    batch_expectations = []

    reweighted = functools.partial(
        target_log_prob_fn,
        data=data,
        prior_weight=jnp.astype(batch_size / dataset_size, jnp.float64),
    )

    def sample_expected_elbo(q_samples, q_lp):

        penalized_ll = target_log_prob_fn(
            data=data,
            prior_weight= batch_size / dataset_size,
            **q_samples,
        )
        expected_elbo = jnp.mean(q_lp * batch_size / dataset_size - penalized_ll)

        return expected_elbo

    for _ in range(sample_batches):
        q_samples, q_lp = sample_elbo()
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
                    # Log-prob is only used if `gradient_estimator == SCORE_FUNCTION`.
                    log_prob=surrogate_posterior.log_prob,
                    use_reparameterization=(
                        gradient_estimator != GradientEstimators.SCORE_FUNCTION
                    ),
                )
            ]
        else:
            batch_expectations += [sample_expected_elbo(q_samples, q_lp)]
    batch_expectations = jnp.mean(batch_expectations, axis=0)
    return batch_expectations


def minibatch_fit_surrogate_posterior(
    target_log_prob_fn,
    surrogate_posterior,
    data_iterator,
    dataset_size: int,
    batch_size: int = 1,
    steps_per_epoch: int = 1,
    num_epochs: int = 1,
    accumulation_steps: int = 1,
    sample_size=8,
    sample_batches=1,
    lr_decay_factor: float = 0.5,
    learning_rate=1.0,
    patience: int = 3,
    trainable_variables: typing.List[str] = None,
    name=None,
    test_fn=None,
    **kwargs,
):
    if trainable_variables is None:
        trainable_variables = list(surrogate_posterior.parameters['model'].keys())

    def complete_variational_loss_fn(data=None):
        """This becomes the loss function called in the
        optimization loop. It gets called on each minibatch of data.
        """
        return minibatch_mc_variational_loss(
            target_log_prob_fn,
            surrogate_posterior,
            sample_size=sample_size,
            sample_batches=sample_batches,
            data=data,
            dataset_size=dataset_size,
            batch_size=batch_size,
            name=name,
            **kwargs,
        )

    return training_loop(
        loss_fn=complete_variational_loss_fn,
        base_optimizer_fn=None,
        trainable_filter=None,
        params=None,
        data_iterator=data_iterator,
        num_epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        accumulation_steps=accumulation_steps,
        learning_rate=learning_rate,
        # trainable_variables=trainable_variables,
        # clip_value=clip_value,
        # clip_by=clip_by,
        patience=patience,
        lr_decay_factor=lr_decay_factor,
        **kwargs,
    )
