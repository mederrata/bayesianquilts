"""Pareto smoothed importance sampling (PSIS) in JAX

This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out (LOO) cross-validation in JAX.

This code is a conversion of the original NumPy implementation by Aki Vehtari
and Tuomas Sivula.

Included functions
------------------
psisloo
    Pareto smoothed importance sampling leave-one-out log predictive densities.

psislw
    Pareto smoothed importance sampling.

gpdfitnew
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).

gpinv
    Inverse Generalised Pareto distribution function.

References
----------
Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Practical
Bayesian model evaluation using leave-one-out cross-validation
and WAIC. Statistics and Computing, 27(5):1413–1432.
doi:10.1007/s11222-016-9696-4. https://arxiv.org/abs/1507.04544

Aki Vehtari, Andrew Gelman and Jonah Gabry (2017). Pareto
smoothed importance sampling. https://arxiv.org/abs/arXiv:1507.02646v5

"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# 3-Clause BSD License
"""
Copyright 2017 Aki Vehtari, Tuomas Sivula

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. """


def psisloo(log_lik, **kwargs):
    r"""PSIS leave-one-out log predictive densities.

    Computes the log predictive densities given posterior samples of the log
    likelihood terms :math:`p(y_i|\theta^s)` in input parameter `log_lik`.
    Returns a sum of the leave-one-out log predictive densities `loo`,
    individual leave-one-out log predictive density terms `loos` and an estimate
    of Pareto tail indeces `ks`. The estimates are unreliable if tail index
    ``k > 0.7``.

    Additional keyword arguments are passed to the :meth:`psislw()` function.

    Parameters
    ----------
    log_lik : jax.Array
        Array of size n x m containing n posterior samples of the log likelihood
        terms :math:`p(y_i|\theta^s)`.

    Returns
    -------
    loo : scalar
        sum of the leave-one-out log predictive densities
    loos : jax.Array
        individual leave-one-out log predictive density terms
    ks : jax.Array
        estimated Pareto tail indeces

    """
    # log raw weights from log_lik
    lw = -log_lik
    # compute Pareto smoothed log weights given raw log weights
    lw_smoothed, ks = psislw(lw, **kwargs)
    # compute leave-one-out log predictive densities
    lw_new = lw_smoothed + log_lik
    loos = logsumexp(lw_new, axis=0)
    loo = loos.sum()
    return loo, loos, ks


def psislw(lw, Reff=1.0):
    """Pareto smoothed importance sampling (PSIS).

    NOTE: This JAX implementation is a direct conversion of the original NumPy
    code. It uses a Python-level for-loop to iterate over the columns of `lw`,
    which means it is NOT JIT-COMPILABLE. The dynamic nature of tail-finding
    makes vectorization with `vmap` or compilation with `jit` non-trivial.

    Parameters
    ----------
    lw : jax.Array
        Array of size n x m containing m sets of n log weights. It is also
        possible to provide one dimensional array of length n.

    Reff : scalar, optional
        relative MCMC efficiency ``N_eff / N``

    Returns
    -------
    lw_out : jax.Array
        smoothed log weights
    kss : jax.Array
        Pareto tail indices

    """
    if lw.ndim == 2:
        n, m = lw.shape
    elif lw.ndim == 1:
        n = len(lw)
        m = 1
        lw = lw.reshape(n, 1)
    else:
        raise ValueError("Argument `lw` must be 1 or 2 dimensional.")
    if n <= 1:
        raise ValueError("More than one log-weight needed.")

    # allocate new array for output
    lw_out = jnp.copy(lw)

    # allocate output array for kss
    kss = jnp.empty(m)

    # precalculate constants
    cutoff_ind = -int(jnp.ceil(min(0.2 * n, 3 * jnp.sqrt(n / Reff)))) - 1
    cutoffmin = jnp.log(jnp.finfo(jnp.float32).tiny)
    k_min = 1/3

    # loop over sets of log weights
    # This loop is in Python and not JAX-transformable (e.g. jax.jit)
    # because the size of the tail (`n2`) is data-dependent.
    for i in range(m):
        x = lw_out[:, i]
        # improve numerical accuracy
        x = x - jnp.max(x)
        # sort the array
        x_sort_ind = jnp.argsort(x)
        # divide log weights into body and right tail
        xcutoff = jnp.maximum(x[x_sort_ind[cutoff_ind]], cutoffmin)
        expxcutoff = jnp.exp(xcutoff)

        tail_mask = x > xcutoff
        n2 = jnp.sum(tail_mask)

        if n2 <= 4:
            # not enough tail samples for gpdfitnew
            k = jnp.inf
            smoothed_tail = None
        else:
            # order of tail samples
            # JAX requires functional updates, so we extract the tail,
            # sort it, and then fit the GPD.
            tail_values = x[tail_mask]
            tail_sort_indices = jnp.argsort(tail_values)
            tail_values_sorted = tail_values[tail_sort_indices]

            # fit generalized Pareto distribution to the right tail samples
            tail_to_fit = jnp.exp(tail_values_sorted) - expxcutoff
            k, sigma = gpdfitnew(tail_to_fit, sort=False)

        if k >= k_min and not jnp.isinf(k):
            # compute ordered statistic for the fit
            sti = (jnp.arange(0.5, n2)) / n2
            qq = gpinv(sti, k, sigma)
            qq = qq + expxcutoff
            qq = jnp.log(qq)

            # Find original indices of the sorted tail to update `x`
            original_tail_indices = jnp.where(tail_mask)[0]
            original_indices_of_sorted_tail = original_tail_indices[tail_sort_indices]

            # place the smoothed tail into the output array
            x = x.at[original_indices_of_sorted_tail].set(qq)
            # truncate smoothed values to the largest raw weight (0)
            x = jnp.minimum(x, 0.0)

        # renormalize weights
        x = x - logsumexp(x)
        # store tail index k
        kss = kss.at[i].set(k)
        lw_out = lw_out.at[:, i].set(x)

    # If the provided input array is one dimensional, return kss as scalar.
    if lw.shape[1] == 1 and lw.ndim == 2:
        kss = kss[0]
        lw_out = lw_out.flatten()

    return lw_out, kss


def gpdfitnew(x, sort=True, return_quadrature=False):
    """Estimate the paramaters for the Generalized Pareto Distribution (GPD)

    Returns empirical Bayes estimate for the parameters of the two-parameter
    generalized Parato distribution given the data.

    NOTE: To make this function JIT-compatible, the step that filters out
    negligible weights in the original NumPy implementation has been removed.
    This should have a minimal impact on the result. The `sort_in_place`
    argument has been removed as JAX arrays are immutable.

    Parameters
    ----------
    x : jax.Array
        One dimensional data array
    sort : bool or jax.Array, optional
        If known in advance, one can provide an array of indices that would
        sort the input array `x`. If the input array is already sorted, provide
        False. If True (default behaviour), the array is sorted internally.
    return_quadrature : bool, optional
        If True, quadrature points and weights `ks` and `w` of the marginal posterior of k are also returned.

    Returns
    -------
    k, sigma : float
        estimated parameter values
    ks, w : jax.Array
        Quadrature points and weights of the marginal posterior distribution
        of `k`. Returned only if `return_quadrature` is True.
    """
    if x.ndim != 1 or len(x) <= 1:
        raise ValueError("Invalid input array.")

    # check if x should be sorted
    if isinstance(sort, bool):
        if sort:
            sort_indices = jnp.argsort(x)
            x_sorted = x[sort_indices]
        else: # array is pre-sorted
            x_sorted = x
    else: # sort is an array of indices
        x_sorted = x[sort]


    n = len(x)
    PRIOR = 3
    m = 30 + int(jnp.sqrt(n))

    i = jnp.arange(1, m + 1, dtype=jnp.float32)
    bs = 1 - jnp.sqrt(m / (i - 0.5))
    bs = bs / (PRIOR * x_sorted[n // 4]) + 1 / x_sorted[-1]


    ks = -bs
    temp = jnp.log1p(ks[:, None] * x)
    ks = jnp.mean(temp, axis=1)

    L = n * (jnp.log(-bs / ks) - ks - 1)

    temp = L - L[:, None]
    w = 1.0 / jnp.sum(jnp.exp(temp), axis=1)

    # NOTE: In the original numpy code, negligible weights were filtered out.
    # To make this function JIT-compatible, we skip that step and just normalize.
    # The effect should be negligible.
    w = w / w.sum()

    # posterior mean for b
    b = jnp.sum(bs * w)
    # Estimate for k
    temp = jnp.log1p(-b * x)
    k = jnp.mean(temp)
    # estimate for sigma
    sigma = -k / b
    # weakly informative prior for k
    a = 10
    k = k * n / (n + a) + a * 0.5 / (n + a)

    if return_quadrature:
        temp = jnp.log1p(-bs[:, None] * x)
        ks_quad = jnp.mean(temp, axis=1)
        ks_quad = ks_quad * n / (n+a) + a * 0.5 / (n+a)
        return k, sigma, ks_quad, w
    else:
        return k, sigma


def gpinv(p, k, sigma):
    """Inverse Generalised Pareto distribution function."""
    # Ensure p is a JAX array
    p = jnp.asarray(p)
    
    # Use lax.cond for scalar conditional on k for JIT compatibility
    def k_is_zero_fun(_):
        return -jnp.log1p(-p)

    def k_is_not_zero_fun(_):
        return jnp.expm1(-k * jnp.log1p(-p)) / k

    x = jax.lax.cond(
        jnp.abs(k) < jnp.finfo(jnp.float32).eps,
        k_is_zero_fun,
        k_is_not_zero_fun,
        operand=None,
    )
    x = x * sigma

    # Boundary conditions
    x = jnp.where(p == 0, 0.0, x)
    val_at_1 = jnp.where(k >= 0, jnp.inf, -sigma / k)
    x = jnp.where(p == 1, val_at_1, x)

    # Final check on sigma
    return jnp.where(sigma <= 0, jnp.full_like(p, jnp.nan), x)