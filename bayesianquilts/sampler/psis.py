import tensorflow as tf


def sumlogs(x, axis=None):
    """Sum of vector where numbers are represented by their logarithms.

    Calculates ``tf.math.log(tf.reduce_sum(tf.exp(x), axis=axis))`` in such a
    fashion that it works even when elements have large magnitude.

    """
    maxx = tf.reduce_max(x, axis=axis, keepdims=True)
    xnorm = x - maxx
    xnorm = tf.exp(xnorm)
    out = tf.reduce_sum(xnorm, axis=axis)
    out = tf.math.log(out)
    out += tf.squeeze(maxx)
    return out

def gpinv(p, k, sigma):
    """Inverse Generalized Pareto distribution function."""
    x = tf.fill(tf.shape(p), tf.constant(float('nan')))
    
    if sigma <= 0:
        return x

    ok = tf.logical_and(tf.greater(p, 0), tf.less(p, 1))

    if tf.reduce_all(ok):
        if tf.abs(k) < tf.constant(tf.finfo(float).eps):
            x = -tf.math.log1p(-p)
        else:
            x = -tf.math.log1p(-p) * k
            x = tf.math.expm1(x) / k
        x *= sigma
    else:
        if tf.abs(k) < tf.constant(tf.finfo(float).eps):
            temp = -tf.math.log1p(-p[ok])
            x = tf.tensor_scatter_nd_update(x, tf.where(ok)[:, tf.newaxis], temp)
        else:
            temp = -tf.math.log1p(-p[ok]) * k
            temp = tf.math.expm1(temp) / k
            x = tf.tensor_scatter_nd_update(x, tf.where(ok)[:, tf.newaxis], temp)
        x *= sigma
        x = tf.tensor_scatter_nd_update(x, tf.where(tf.equal(p, 0))[:, tf.newaxis], tf.zeros_like(tf.where(tf.equal(p, 0))))
        if k >= 0:
            x = tf.tensor_scatter_nd_update(x, tf.where(tf.equal(p, 1))[:, tf.newaxis], tf.fill(tf.shape(tf.where(tf.equal(p, 1))), tf.constant(float('inf'))))
        else:
            x = tf.tensor_scatter_nd_update(x, tf.where(tf.equal(p, 1))[:, tf.newaxis], -sigma / k)

    return x

def psisloo(log_lik, **kwargs):
    r"""PSIS leave-one-out log predictive densities.

    Computes the log predictive densities given posterior samples of the log
    likelihood terms :math:`p(y_i|\theta^s)` in input parameter `log_lik`.
    Returns a sum of the leave-one-out log predictive densities `loo`,
    individual leave-one-out log predictive density terms `loos` and an estimate
    of Pareto tail indices `ks`. The estimates are unreliable if tail index
    ``k > 0.7`` (see more in the references listed in the module docstring).

    Additional keyword arguments are passed to the :meth:`psislw_tf()` function
    (see the corresponding documentation).

    Parameters
    ----------
    log_lik : tf.Tensor
        Tensor of shape (n, m) containing n posterior samples of the log likelihood
        terms :math:`p(y_i|\theta^s)`.

    Returns
    -------
    loo : scalar
        Sum of the leave-one-out log predictive densities.

    loos : tf.Tensor
        Individual leave-one-out log predictive density terms.

    ks : tf.Tensor
        Estimated Pareto tail indices.

    """
    # Ensure overwrite flag is passed in the arguments
    kwargs['overwrite_lw'] = True
    # Log raw weights from log_lik
    lw = -log_lik
    # Compute Pareto smoothed log weights given raw log weights
    lw, ks = psislw(lw, **kwargs)
    # Compute
    lw += log_lik
    loos = sumlogs(lw, axis=0)
    loo = tf.reduce_sum(loos)
    return loo, loos, ks

def psislw(lw, Reff=1.0, overwrite_lw=False):
    """Pareto smoothed importance sampling (PSIS).

    Parameters
    ----------
    lw : tf.Tensor
        Tensor of shape (n, m) containing m sets of n log weights. It is also
        possible to provide a one-dimensional tensor of length n.

    Reff : float, optional
        Relative MCMC efficiency ``N_eff / N``

    overwrite_lw : bool, optional
        If True, the input tensor `lw` is smoothed in-place, assuming the tensor
        is F-contiguous. By default, a new tensor is allocated.

    Returns
    -------
    lw_out : tf.Tensor
        Smoothed log weights.

    kss : tf.Tensor
        Pareto tail indices.

    """
    if lw.shape.ndims == 2:
        n, m = lw.shape
    elif lw.shape.ndims == 1:
        n = tf.shape(lw)[0]
        m = 1
    else:
        raise ValueError("Argument `lw` must be 1 or 2 dimensional.")
    
    if n <= 1:
        raise ValueError("More than one log-weight needed.")

    if overwrite_lw and tf.compat.dimension_value(lw.shape[-1]) == m and lw.dtype.is_floating:
        # in-place operation
        lw_out = lw
    else:
        # allocate new tensor for output
        lw_out = tf.identity(lw)

    # allocate output tensor for kss
    kss = tf.TensorArray(dtype=tf.float64, size=m)

    # precalculate constants
    cutoff_ind = -tf.cast(tf.math.ceil(tf.math.minimum(0.2 * tf.cast(n, tf.float32), 3 * tf.math.sqrt(n / Reff))), tf.int32) - 1
    cutoffmin = tf.cast(-38, tf.float64)
    logn = tf.math.log(tf.cast(n, tf.float64))
    k_min = 1/3

    # loop over sets of log weights
    for i in range(m):
        x = lw_out[:, i] if m > 1 else lw_out

        # improve numerical accuracy
        x -= tf.reduce_max(x)
        # sort the tensor
        x_sort_ind = tf.argsort(x)
        # divide log weights into body and right tail
        xcutoff = tf.maximum(x[x_sort_ind[cutoff_ind]], cutoffmin)
        expxcutoff = tf.exp(xcutoff)
        tailinds = tf.where(x > xcutoff)[:, 0]
        x2 = tf.gather(x, tailinds)
        n2 = tf.shape(x2)[0]

        def tail_smoothing(x2):
            # fit generalized Pareto distribution to the right tail samples
            x2 = tf.exp(x2) - expxcutoff
            x2si = tf.argsort(x2)
            k, sigma = gpdfitnew(x2, sort=x2si)

            # no smoothing if short tail or GPD fit failed
            if k < k_min or tf.math.is_inf(k):
                return x

            # compute ordered statistic for the fit
            sti = tf.range(0.5, tf.cast(n2, dtype=tf.float64))
            sti /= tf.cast(n2, dtype=tf.float64)
            qq = gpinv(sti, k, sigma)
            qq += expxcutoff
            qq = tf.math.log(qq)

            # place the smoothed tail into the output tensor
            x = tf.tensor_scatter_nd_update(x, tf.expand_dims(tailinds[x2si], axis=1), tf.expand_dims(qq, axis=1))

            # truncate smoothed values to the largest raw weight 0
            x = tf.where(x > 0, 0.0, x)

            return x

        x = tf.cond(tf.less_equal(n2, 4), lambda: x, tail_smoothing(x2))

        # renormalize weights
        x -= sumlogs(x)
        # store tail index k
        kss = kss.write(i, k)

    # If the provided input tensor is one dimensional, return kss as scalar.
    if m == 1:
        kss = kss.stack()[0]

    return lw_out, kss


def gpdfitnew(x, sort=True, sort_in_place=False, return_quadrature=False):
    """Estimate the parameters for the Generalized Pareto Distribution (GPD)

    Returns empirical Bayes estimate for the parameters of the two-parameter
    generalized Pareto distribution given the data.

    Parameters
    ----------
    x : tf.Tensor
        One-dimensional data tensor.

    sort : bool or tf.Tensor, optional
        If known in advance, one can provide a tensor of indices that would
        sort the input tensor `x`. If the input tensor is already sorted, provide
        False. If True (default behavior), the tensor is sorted internally.

    sort_in_place : bool, optional
        If `sort` is True and `sort_in_place` is True, the tensor is sorted
        in-place (False by default).

    return_quadrature : bool, optional
        If True, quadrature points and weights `ks` and `w` of the marginal
        posterior distribution of k are also calculated and returned. False by
        default.

    Returns
    -------
    k, sigma : tf.Tensor
        Estimated parameter values.

    ks, w : tf.Tensor
        Quadrature points and weights of the marginal posterior distribution
        of `k`. Returned only if `return_quadrature` is True.

    Notes
    -----
    This function returns a negative of Zhang and Stephens's k, because it is
    more common parameterization.

    """
    if x.shape.ndims != 1 or tf.size(x) <= 1:
        raise ValueError("Invalid input tensor.")

    # check if x should be sorted
    if sort is True:
        if sort_in_place:
            x = tf.sort(x)
            xsorted = True
        else:
            sort = tf.argsort(x)
            xsorted = False
    elif sort is False:
        xsorted = True
    else:
        xsorted = False

    n = tf.size(x)
    PRIOR = 3
    m = 30 + tf.cast(tf.sqrt(tf.cast(n, dtype=tf.float64)), dtype=tf.int32)

    bs = tf.range(1.0, tf.cast(m + 1, dtype=tf.float64))
    bs -= 0.5
    bs /= tf.sqrt(bs)
    bs = 1 - bs
    if xsorted:
        bs /= PRIOR * x[tf.cast(n/4 + 0.5, dtype=tf.int32) - 1]
        bs += 1 / x[-1]
    else:
        bs /= PRIOR * x[sort[tf.cast(n/4 + 0.5, dtype=tf.int32) - 1]]
        bs += 1 / x[sort[-1]]

    ks = -bs
    temp = ks[:, None] * x
    temp = tf.math.log1p(temp)
    ks = tf.reduce_mean(temp, axis=1)

    L = bs / ks
    L = -L
    L = tf.math.log(L)
    L -= ks
    L -= 1
    L *= tf.cast(n, dtype=tf.float64)

    temp = L - tf.transpose(L)
    temp = tf.exp(temp)
    w = tf.reduce_sum(temp, axis=1)
    w = 1 / w

    # remove negligible weights
    dii = w >= 10 * tf.constant(tf.finfo(float).eps, dtype=tf.float64)
    if not tf.reduce_all(dii):
        w = tf.boolean_mask(w, dii)
        bs = tf.boolean_mask(bs, dii)
    # normalize w
    w /= tf.reduce_sum(w)

    # posterior mean for b
    b = tf.reduce_sum(bs * w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterization.
    temp = (-b) * x
    temp = tf.math.log1p(temp)
    k = tf.reduce_mean(temp)
    if return_quadrature:
        temp = -x
        temp = bs[:, None] * temp
        temp = tf.math.log1p(temp)
        ks = tf.reduce_mean(temp, axis=1)
    # estimate for sigma
    sigma = -k / b * tf.cast(n, dtype=tf.float64) / (tf.cast(n, dtype=tf.float64) - 0)
    # weakly informative prior for k
    a = 10.0
    k = k * tf.cast(n, dtype=tf.float64) / (tf.cast(n, dtype=tf.float64) + a) + a * 0.5 / (tf.cast(n, dtype=tf.float64) + a)
    if return_quadrature:
        ks *= tf.cast(n, dtype=tf.float64) / (tf.cast(n, dtype=tf.float64) + a)
        ks += a * 0.5 / (tf.cast(n, dtype=tf.float64) + a)

    if return_quadrature:
        return k, sigma, ks, w
    else:
        return k, sigma