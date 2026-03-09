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


def _unwrap_dist(dist):
    """Unwrap a TransformedDistribution to get (base_dist, bijector).

    If `dist` is a TransformedDistribution, returns (base, bijector).
    Otherwise returns (dist, None).  Used by QMC sampling to access
    the underlying Normal's loc/scale.
    """
    import tensorflow_probability.substrates.jax.distributions as tfd
    if isinstance(dist, tfd.TransformedDistribution):
        return dist.distribution, dist.bijector
    return dist, None


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
    kl_weight: float = 1.0,
    qmc_base: jnp.ndarray | None = None,
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
        cost: "reweighted" (default), "tfp" for TFP's importance-weighted
            divergence, or "iwae" for the IWAE bound with DReG gradients
            (Tucker et al., 2019). When cost="iwae", sample_size controls
            the number of importance samples K. The STL flag is ignored
            for IWAE since DReG provides its own low-variance gradient
            estimator. K=1 reduces to the standard ELBO.
        kl_weight: Weight for the KL divergence term (default 1.0). Used for
            KL annealing: ramp from 0 to 1 over early epochs to prevent
            posterior collapse. The loss becomes:
            kl_weight * E_q[log q - log p(theta)] - (N/B) * E_q[log p(batch | theta)]
        qmc_base: Pre-generated Sobol base points of shape (sample_size, total_dim).
            When provided, replaces iid normal samples with randomized QMC
            (scrambled Sobol + random shift + inverse normal CDF) for ~2x
            variance reduction. Generated outside JIT in
            minibatch_fit_surrogate_posterior.

    Returns:
        Scalar per-datum loss value (negative ELBO / N).
    """
    import tensorflow_probability.substrates.jax.distributions as tfd

    scale = dataset_size / batch_size

    def sample_elbo(key):
        if qmc_base is not None:
            return _qmc_sample_elbo(key, surrogate_posterior, qmc_base, sample_size)
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
        # With kl_weight:
        #   loss = kl_weight * q_lp - scale * target(prior_weight=kl_weight*B/N)
        # When kl_weight=1.0 this is the standard ELBO.
        penalized_ll = target_log_prob_fn(
            data=data,
            prior_weight=kl_weight * batch_size / dataset_size,
            **q_samples,
        )
        # STL: stop gradient through log q(theta) to reduce variance
        q_lp_grad = jax.lax.stop_gradient(q_lp) if stl else q_lp
        expected_elbo = jnp.mean(kl_weight * q_lp_grad - scale * penalized_ll)
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
        elif cost == "iwae":
            # IWAE bound with DReG gradient estimator (Tucker et al., 2019).
            # log_w_k = log p(x, z_k) - log q(z_k)  for k = 1..K
            # IWAE = log (1/K) sum_k exp(log_w_k)
            # DReG: differentiate through squared normalized weights for
            # unbiased, low-variance gradients.
            penalized_ll = target_log_prob_fn(
                data=data,
                prior_weight=kl_weight * batch_size / dataset_size,
                **q_samples,
            )
            log_w = scale * penalized_ll - kl_weight * q_lp  # (K,)

            # DReG: stop-gradient the normalized weights, square them,
            # and use as surrogate loss coefficients
            log_w_stopped = jax.lax.stop_gradient(log_w)
            w_norm = jax.nn.softmax(log_w_stopped)
            dreg_loss = -jnp.sum(jax.lax.stop_gradient(w_norm ** 2) * log_w)

            batch_expectations += [dreg_loss]
            batch_expectations = jnp.atleast_1d(batch_expectations)
        else:
            batch_expectations += [sample_expected_elbo(q_samples, q_lp)]
            batch_expectations = jnp.atleast_1d(batch_expectations)
    batch_expectations = jnp.mean(batch_expectations, axis=0)
    return batch_expectations / dataset_size


def _qmc_sample_elbo(key, surrogate_posterior, qmc_base, sample_size):
    """Generate QMC samples from the surrogate posterior.

    Applies a random digital shift to the pre-generated Sobol base points,
    transforms through the inverse normal CDF, and reparameterizes through
    each variable's loc/scale.

    Args:
        key: JAX PRNG key for the random shift.
        surrogate_posterior: TFP JointDistributionNamed or CrossVariableMVN.
        qmc_base: Sobol base points, shape (sample_size, total_dim).
        sample_size: Number of samples K.

    Returns:
        (samples_dict, q_lp) tuple.
    """
    import tensorflow_probability.substrates.jax.distributions as tfd

    shift = jax.random.uniform(key, shape=(qmc_base.shape[-1],))
    shifted = (qmc_base + shift[None, :]) % 1.0
    normals = jax.scipy.special.ndtri(jnp.clip(shifted, 1e-6, 1 - 1e-6))

    samples = {}
    offset = 0
    for name, dist in surrogate_posterior.model.items():
        base, bij = _unwrap_dist(dist)
        flat_dim = int(np.prod(base.event_shape))
        eps = normals[:, offset:offset + flat_dim]

        if isinstance(base, tfd.Independent) and isinstance(
            base.distribution, tfd.Normal
        ):
            z = base.distribution.loc + base.distribution.scale * eps.reshape(
                sample_size, *base.distribution.loc.shape
            )
        elif isinstance(base, tfd.MultivariateNormalTriL):
            z = base.loc + eps @ base.scale_tril.T
        else:
            # Fallback to iid for non-Normal (e.g. InverseGamma)
            key, sk = jax.random.split(key)
            samples[name] = dist.sample(sample_size, seed=sk)
            continue

        if bij is not None:
            z = bij.forward(z)
        samples[name] = z
        offset += flat_dim

    q_lp = surrogate_posterior.log_prob(samples)
    return samples, q_lp


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
    kl_anneal_epochs: int = 0,
    qmc: bool = False,
    **kwargs,
):
    if initial_values is None:
        initial_values = surrogate_initializer()

    # Set up PRNG key for reproducible sampling
    if seed is not None:
        base_key = random.PRNGKey(seed)
    else:
        base_key = None

    # Pre-generate Sobol base points for QMC (outside JIT)
    qmc_base = None
    if qmc:
        from scipy.stats import qmc as scipy_qmc
        test_surrogate = surrogate_generator(initial_values)
        total_dim = sum(
            int(np.prod(d.event_shape))
            for d in test_surrogate.model.values()
        )
        n_pow2 = int(2 ** np.ceil(np.log2(max(sample_size, 2))))
        sobol = scipy_qmc.Sobol(d=total_dim, scramble=False)
        qmc_base = jnp.array(sobol.random(n_pow2)[:sample_size])

    # KL annealing: track step count via mutable closure
    step_counter = [0]
    total_anneal_steps = kl_anneal_epochs * steps_per_epoch

    def complete_variational_loss_fn(data=None, params=None, seed=None):
        """Loss function called in the optimization loop on each minibatch."""
        if params is None:
            params = initial_values

        # Compute KL weight for annealing
        if total_anneal_steps > 0:
            kl_weight = jnp.minimum(1.0, step_counter[0] / total_anneal_steps)
            step_counter[0] += 1
        else:
            kl_weight = 1.0

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
            kl_weight=kl_weight,
            qmc_base=qmc_base,
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
