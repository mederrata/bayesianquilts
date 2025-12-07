
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import Distribution

def factorized_distribution_moments(
    distribution: Dict[str, Distribution], 
    samples: int = 250, 
    exclude: List[str] = [],
    seed: jax.random.PRNGKey = None
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Compute the mean and variance of a factorized distribution.

    Args:
        distribution (Dict[str, Distribution]): A dictionary of distributions.
        samples (int, optional): The number of samples to use if the mean and
            variance are not implemented. Defaults to 250.
        exclude (List[str], optional): A list of distribution names to exclude.
            Defaults to [].
        seed (jax.random.PRNGKey, optional): A random key for sampling.
            Defaults to None.

    Returns:
        Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]: A tuple of
            dictionaries containing the means and variances of the distributions.
    """
    means = {}
    variances = {}
    if seed is None:
        seed = jax.random.PRNGKey(0)

    for k, dist in distribution.items():
        if k in exclude:
            continue
        if not isinstance(dist, Distribution):
            raise AttributeError("Need a TFP distribution object")

        try:
            mean = dist.mean()
            variance = dist.variance()
        except NotImplementedError:
            def sample_and_square(key):
                s = dist.sample(seed=key)
                return s, s**2
            
            keys = jax.random.split(seed, samples)
            sum_1, sum_2 = jax.vmap(sample_and_square)(keys)
            
            mean = jnp.mean(sum_1, axis=0)
            variance = jnp.mean(sum_2, axis=0) - mean ** 2

        means[k] = mean
        variances[k] = variance
    return means, variances
