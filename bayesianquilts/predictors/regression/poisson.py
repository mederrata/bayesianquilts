import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import Dict, Any
from bayesianquilts.model import BayesianModel
from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin


class PoissonRegression(BayesianModel):
    def __init__(
        self,
        input_dim,
        dtype=jnp.float64,
        prior_scale_beta=10.0,
        prior_scale_intercept=10.0,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.prior_scale_beta = prior_scale_beta
        self.prior_scale_intercept = prior_scale_intercept
        self.var_list = ["beta", "intercept"]
        self.create_distributions()

    def create_distributions(self):
        # Surrogate: Mean-field Normal
        self.prior_distribution = tfd.JointDistributionNamed(
            {
                "beta": tfd.Independent(
                    tfd.Normal(
                        jnp.zeros(self.input_dim, dtype=self.dtype),
                        jnp.array(self.prior_scale_beta, dtype=self.dtype),
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                "intercept": tfd.Independent(
                    tfd.Normal(
                        jnp.zeros(1, dtype=self.dtype),
                        jnp.array(self.prior_scale_intercept, dtype=self.dtype),
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            }
        )

    def surrogate_parameter_initializer(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key, 2)

        def init_mean_scale(shape, k):
            ska, skb = jax.random.split(k)
            mean = jax.random.normal(ska, shape, dtype=self.dtype) * 0.01
            raw_scale = (
                jnp.log(jnp.exp(0.01) - 1.0)
                + jax.random.normal(skb, shape, dtype=self.dtype) * 0.001
            )
            return mean, raw_scale

        beta_loc, beta_scale = init_mean_scale((self.input_dim,), k1)
        int_loc, int_scale = init_mean_scale((1,), k2)

        return {
            "beta_loc": beta_loc,
            "beta_raw_scale": beta_scale,
            "intercept_loc": int_loc,
            "intercept_raw_scale": int_scale,
        }

    def surrogate_distribution_generator(self, params):
        return tfd.JointDistributionNamed(
            {
                "beta": tfd.Independent(
                    tfd.Normal(
                        params["beta_loc"],
                        jax.nn.softplus(params["beta_raw_scale"]) + 1e-5,
                    ),
                    reinterpreted_batch_ndims=1,
                ),
                "intercept": tfd.Independent(
                    tfd.Normal(
                        params["intercept_loc"],
                        jax.nn.softplus(params["intercept_raw_scale"]) + 1e-5,
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            }
        )

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        X = data["X"]
        y = data["y"]
        offset = data.get("offset", jnp.zeros_like(y))

        beta = params["beta"]
        intercept = params["intercept"]

        has_sample_dim = beta.ndim > 1

        # Priors
        log_prior = self.prior_distribution.log_prob(params)

        if has_sample_dim:
            # Likelihood
            # X: (B, D), beta: (S, D) -> (S, B)
            eta = jnp.einsum("bd,sd->sb", X, beta) + intercept + offset
        else:
            eta = jnp.dot(X, beta) + intercept + offset

        rate = jnp.exp(eta)

        # Poisson Log Likelihood
        if has_sample_dim:
            log_lik = tfd.Poisson(rate=rate).log_prob(y)
            log_lik = jnp.sum(log_lik, axis=-1)
        else:
            log_lik = jnp.sum(tfd.Poisson(rate=rate).log_prob(y))

        return log_lik + log_prior * prior_weight

    def log_likelihood(self, data, **params):
        X = data["X"]
        y = data["y"]
        offset = data.get("offset", jnp.zeros_like(y))
        beta = params["beta"]
        intercept = params["intercept"]

        has_sample_dim = beta.ndim > 1

        if has_sample_dim:
            eta = jnp.einsum("bd,sd->sb", X, beta) + intercept + offset
        else:
            eta = jnp.dot(X, beta) + intercept + offset

        rate = jnp.exp(eta)
        # Returns (S, B) or (B,)
        return tfd.Poisson(rate=rate).log_prob(y)

    def predictive_distribution(self, data, **params):
        X = data["X"]
        y = data["y"]
        offset = data.get("offset", jnp.zeros_like(y))
        beta = params["beta"]
        intercept = params["intercept"]

        has_sample_dim = beta.ndim > 1
        if has_sample_dim:
            eta = jnp.einsum("bd,sd->sb", X, beta) + intercept + offset
        else:
            eta = jnp.dot(X, beta) + intercept + offset

        rate = jnp.exp(eta)
        return {"prediction": rate}


class PoissonRegressionLikelihood(AutoDiffLikelihoodMixin):
    """Likelihood function for Poisson regression."""

    def __init__(self, dtype=jnp.float64):
        self.dtype = dtype

    def log_likelihood(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute log-likelihood for Poisson regression."""
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)  # counts
        y = jnp.atleast_1d(jnp.squeeze(y))
        offset = data.get("offset", jnp.zeros_like(y))

        beta = params["beta"]
        intercept = params["intercept"]
        
        # Handle accumulated dimensions from chained transformations
        # Both beta and intercept can accumulate dimensions
        

        
        # Proceed with existing logic (simplified)
        
        # Squeeze beta down to at most 3D (S, N, F) or 2D (S, F)
        while beta.ndim > 3:
            beta = jnp.squeeze(
                beta,
                axis=tuple(i for i in range(1, beta.ndim - 1) if beta.shape[i] == 1),
            )
            if beta.ndim > 3:
                # Fallback: just take first slice
                beta = beta[:, 0, ...]

        # Squeeze intercept
        while intercept.ndim > 2:
            intercept = jnp.squeeze(
                intercept,
                axis=tuple(
                    i for i in range(1, intercept.ndim - 1) if intercept.shape[i] == 1
                ),
            )
            if intercept.ndim > 2:
                intercept = intercept[:, 0, ...]

        # Final squeeze to remove any trailing singleton dimensions
        intercept = jnp.squeeze(intercept)
        if intercept.ndim == 0:
            intercept = jnp.atleast_1d(intercept)

        # Now beta should be 2D or 3D
        if beta.ndim == 2:
            # Shape (S, F), intercept should be (S,)
            # Check broadcasting for intercept
            if intercept.ndim == 1 and intercept.shape[0] != beta.shape[0]:
                 # Try to broadcast intercept to beta
                 intercept = jnp.broadcast_to(intercept, (beta.shape[0],))
            
            log_rate = jnp.einsum("df,sf->sd", X, beta) + intercept[:, jnp.newaxis] + offset
        elif beta.ndim == 3:
            # Shape (S, N, F)
            # Ensure N matches X
            if beta.shape[1] != X.shape[0]:
                 # Could be mismatch
                 pass

            if intercept.ndim == 1:
                log_rate = jnp.einsum("df,sdf->sd", X, beta) + intercept[:, jnp.newaxis] + offset
            else:
                log_rate = jnp.einsum("df,sdf->sd", X, beta) + intercept + offset
        else:
            raise ValueError(
                f"beta shape {beta.shape} not supported in log_likelihood"
            )

        rate = jnp.exp(log_rate)
        log_lik = y[jnp.newaxis, :] * log_rate - rate

        return log_lik

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
        
        # Ensure we don't have redundant trailing dims of 1 for intercept
        if intercept.ndim > 0 and intercept.shape[-1] == 1:
            intercept = jnp.squeeze(intercept, axis=-1)

        # Broadcast to matching batch shapes if needed
        try:
            target_shape = jnp.broadcast_shapes(beta.shape[:-1], intercept.shape)
            beta = jnp.broadcast_to(beta, target_shape + (beta.shape[-1],))
            intercept = jnp.broadcast_to(intercept, target_shape)
        except ValueError:
            # Fallback if shapes don't broadcast straightforwardly (e.g. mismatch)
            # This shouldn't happen with correct usage but for safety
            pass

        theta = jnp.concatenate([beta, intercept[..., jnp.newaxis]], axis=-1)
        return theta

    def reconstruct_parameters(
        self, flat_params: jnp.ndarray, template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]
        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]
        # intercept needs to be compatible with beta output, usually (...,)
        return {"beta": beta, "intercept": intercept}
