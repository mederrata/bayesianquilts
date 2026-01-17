import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import Dict, Any
from bayesianquilts.model import BayesianModel
from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin


class NegativeBinomialRegression(BayesianModel):
    def __init__(self, input_dim, zero_inflated=False, dtype=jnp.float32, **kwargs):
        """
        Negative Binomial GLM with optional zero-inflation.

        Args:
            input_dim: Number of input features
            zero_inflated: If True, uses zero-inflated negative binomial
            dtype: Data type for computations

        Parameters:
            beta: Regression coefficients (input_dim,)
            intercept: Intercept term (1,)
            log_concentration: Log of concentration parameter (1,)
            zero_logit: Logit of zero-inflation probability (1,) [only if zero_inflated=True]
        """
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.zero_inflated = zero_inflated

        if zero_inflated:
            self.var_list = ["beta", "intercept", "log_concentration", "zero_logit"]
        else:
            self.var_list = ["beta", "intercept", "log_concentration"]

        self.create_distributions()

    def create_distributions(self):
        # Surrogate: Mean-field Normal
        pass  # Using generator

    def surrogate_parameter_initializer(self, key=None, **kwargs):
        if key is None:
            key = jax.random.PRNGKey(42)

        keys = jax.random.split(key, len(self.var_list))

        def init_mean_scale(shape, k):
            ska, skb = jax.random.split(k)
            mean = jax.random.normal(ska, shape, dtype=self.dtype) * 0.01
            raw_scale = (
                jnp.log(jnp.exp(0.01) - 1.0)
                + jax.random.normal(skb, shape, dtype=self.dtype) * 0.001
            )
            return mean, raw_scale

        result = {}

        # Beta
        beta_loc, beta_scale = init_mean_scale((self.input_dim,), keys[0])
        result["beta_loc"] = beta_loc
        result["beta_raw_scale"] = beta_scale

        # Intercept
        int_loc, int_scale = init_mean_scale((1,), keys[1])
        result["intercept_loc"] = int_loc
        result["intercept_raw_scale"] = int_scale

        # Log concentration
        conc_loc, conc_scale = init_mean_scale((1,), keys[2])
        result["log_concentration_loc"] = conc_loc
        result["log_concentration_raw_scale"] = conc_scale

        # Zero logit (if zero-inflated)
        if self.zero_inflated:
            zero_loc, zero_scale = init_mean_scale((1,), keys[3])
            result["zero_logit_loc"] = zero_loc
            result["zero_logit_raw_scale"] = zero_scale

        return result

    def surrogate_distribution_generator(self, params):
        dist_dict = {
            "beta": tfd.Independent(
                tfd.Normal(
                    params["beta_loc"], jax.nn.softplus(params["beta_raw_scale"]) + 1e-5
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
            "log_concentration": tfd.Independent(
                tfd.Normal(
                    params["log_concentration_loc"],
                    jax.nn.softplus(params["log_concentration_raw_scale"]) + 1e-5,
                ),
                reinterpreted_batch_ndims=1,
            ),
        }

        if self.zero_inflated:
            dist_dict["zero_logit"] = tfd.Independent(
                tfd.Normal(
                    params["zero_logit_loc"],
                    jax.nn.softplus(params["zero_logit_raw_scale"]) + 1e-5,
                ),
                reinterpreted_batch_ndims=1,
            )

        return tfd.JointDistributionNamed(dist_dict)

    def _compute_mean_and_concentration(self, X, beta, intercept, log_concentration):
        """Compute mean and concentration parameters."""
        # beta shape: (n_feat,) or (S, n_feat) or (S, N, n_feat)
        # X shape: (N, n_feat)

        if beta.ndim == 1:
            # Case 1: Single sample (n_feat,)
            # X (N, D) @ beta (D) -> (N,)
            eta = jnp.dot(X, beta) + intercept
        elif beta.ndim == 2:
            # Case 2: Standard MCMC/VI (S, n_feat)
            # X (N, D), beta (S, D) -> (S, N)
            eta = jnp.einsum("nd,sd->sn", X, beta) + intercept
        elif beta.ndim == 3:
            # Case 3: AIS per-datum parameters (S, N, n_feat)
            # X (N, D), beta (S, N, D) -> (S, N)
            eta = jnp.einsum("nd,snd->sn", X, beta)
            if intercept.ndim == 3:
                eta = eta + jnp.squeeze(intercept, axis=-1)
            else:
                eta = eta + intercept
        else:
            raise ValueError(f"Unsupported beta dimensionality: {beta.ndim}")

        mean = jnp.exp(eta).astype(self.dtype)

        if log_concentration.ndim == 3:
            concentration = jnp.exp(jnp.squeeze(log_concentration, axis=-1))
        else:
            concentration = jnp.exp(log_concentration)

        concentration = (concentration + jnp.array(1e-6, dtype=self.dtype)).astype(
            self.dtype
        )

        # Convert to NegativeBinomial parameters
        total_count = concentration
        # Broadcast concentration if needed for probs calculation
        if total_count.ndim < mean.ndim:
            total_count_exp = total_count[..., jnp.newaxis]
        else:
            total_count_exp = total_count

        probs = total_count_exp / (total_count_exp + mean)
        probs = jnp.clip(
            probs,
            jnp.array(1e-6, dtype=self.dtype),
            jnp.array(1 - 1e-6, dtype=self.dtype),
        )

        return mean, total_count, probs

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        X = data["X"].astype(self.dtype)
        y = data["y"].astype(self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]

        has_sample_dim = beta.ndim > 1

        # Priors
        # Standardize prior shapes to (S, 1) or ()
        if has_sample_dim:
            p_beta = jnp.sum(
                tfd.Normal(
                    jnp.array(0.0, dtype=self.dtype), jnp.array(10.0, dtype=self.dtype)
                ).log_prob(beta),
                axis=-1,
                keepdims=True,
            )
            p_int = jnp.sum(
                tfd.Normal(
                    jnp.array(0.0, dtype=self.dtype), jnp.array(10.0, dtype=self.dtype)
                ).log_prob(intercept),
                axis=-1,
                keepdims=True,
            )
            p_conc = jnp.sum(
                tfd.Normal(
                    jnp.array(0.0, dtype=self.dtype), jnp.array(2.0, dtype=self.dtype)
                ).log_prob(log_concentration),
                axis=-1,
                keepdims=True,
            )
            log_prior = p_beta + p_int + p_conc
        else:
            log_prior = (
                jnp.sum(
                    tfd.Normal(
                        jnp.array(0.0, dtype=self.dtype),
                        jnp.array(10.0, dtype=self.dtype),
                    ).log_prob(beta)
                )
                + jnp.sum(
                    tfd.Normal(
                        jnp.array(0.0, dtype=self.dtype),
                        jnp.array(10.0, dtype=self.dtype),
                    ).log_prob(intercept)
                )
                + jnp.sum(
                    tfd.Normal(
                        jnp.array(0.0, dtype=self.dtype),
                        jnp.array(2.0, dtype=self.dtype),
                    ).log_prob(log_concentration)
                )
            )

        # Zero-inflation prior
        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            if has_sample_dim:
                p_zero = jnp.sum(
                    tfd.Normal(
                        jnp.array(0.0, dtype=self.dtype),
                        jnp.array(2.0, dtype=self.dtype),
                    ).log_prob(zero_logit),
                    axis=-1,
                    keepdims=True,
                )
                log_prior = log_prior + p_zero
            else:
                log_prior = log_prior + jnp.sum(
                    tfd.Normal(
                        jnp.array(0.0, dtype=self.dtype),
                        jnp.array(2.0, dtype=self.dtype),
                    ).log_prob(zero_logit)
                )

        # Likelihood
        mean, total_count, probs = self._compute_mean_and_concentration(
            X, beta, intercept, log_concentration
        )

        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            zero_prob = jax.nn.sigmoid(zero_logit).astype(self.dtype)

            # Zero-inflated negative binomial
            nb_dist = tfd.NegativeBinomial(total_count=total_count, probs=probs)
            nb_logprob = nb_dist.log_prob(y.astype(self.dtype))
            nb_logprob_zero = nb_dist.log_prob(jnp.zeros_like(y, dtype=self.dtype))

            eps = jnp.array(1e-10, dtype=self.dtype)
            log_zero_prob = jnp.log(zero_prob + eps)
            log_one_minus_zero_prob = jnp.log(
                jnp.array(1.0, dtype=self.dtype) - zero_prob + eps
            )

            ll_zero = jnp.logaddexp(
                log_zero_prob, log_one_minus_zero_prob + nb_logprob_zero
            )
            ll_nonzero = log_one_minus_zero_prob + nb_logprob

            log_lik = jnp.where(y == 0, ll_zero, ll_nonzero)
        else:
            # Standard negative binomial
            log_lik = tfd.NegativeBinomial(
                total_count=total_count, probs=probs
            ).log_prob(y)

        if has_sample_dim:
            log_lik = jnp.sum(log_lik, axis=-1, keepdims=True)
        else:
            log_lik = jnp.sum(log_lik)

        return (log_lik + log_prior * prior_weight).squeeze()

    def log_likelihood(self, data, **params):
        X = data["X"].astype(self.dtype)
        y = data["y"].astype(self.dtype)
        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]

        mean, total_count, probs = self._compute_mean_and_concentration(
            X, beta, intercept, log_concentration
        )

        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            zero_prob = jax.nn.sigmoid(zero_logit).astype(self.dtype)

            nb_dist = tfd.NegativeBinomial(total_count=total_count, probs=probs)
            nb_logprob = nb_dist.log_prob(y.astype(self.dtype))
            nb_logprob_zero = nb_dist.log_prob(jnp.zeros_like(y, dtype=self.dtype))

            eps = jnp.array(1e-10, dtype=self.dtype)
            log_zero_prob = jnp.log(zero_prob + eps)
            log_one_minus_zero_prob = jnp.log(
                jnp.array(1.0, dtype=self.dtype) - zero_prob + eps
            )

            ll_zero = jnp.logaddexp(
                log_zero_prob, log_one_minus_zero_prob + nb_logprob_zero
            )
            ll_nonzero = log_one_minus_zero_prob + nb_logprob

            return jnp.where(y == 0, ll_zero, ll_nonzero).squeeze()
        else:
            return tfd.NegativeBinomial(total_count=total_count, probs=probs).log_prob(
                y.astype(self.dtype)
            )

    def predictive_distribution(self, data, **params):
        X = data["X"].astype(self.dtype)
        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]

        mean, total_count, probs = self._compute_mean_and_concentration(
            X, beta, intercept, log_concentration
        )

        result = {
            "prediction": mean,
            "mean": mean,
            "concentration": (
                jnp.exp(log_concentration) + jnp.array(1e-6, dtype=self.dtype)
            ).astype(self.dtype),
        }

        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            result["zero_prob"] = jax.nn.sigmoid(zero_logit).astype(self.dtype)

        return result


class NegativeBinomialRegressionLikelihood(AutoDiffLikelihoodMixin):
    """Likelihood function for Negative Binomial GLM regression."""

    def __init__(self, model):
        self.model = model
        self.dtype = model.dtype
        self.zero_inflated = model.zero_inflated

    def log_likelihood(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute log-likelihood for Negative Binomial regression."""
        return self.model.log_likelihood(data, **params)

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]

        # Ensure consistent dimensions for concatenation
        # We want (..., K)
        param_list = [beta]

        # Squeeze intercept/conc if they have trailing 1 but match leading dims
        if intercept.ndim > beta.ndim - 1:
            param_list.append(jnp.squeeze(intercept, axis=-1)[..., jnp.newaxis])
        else:
            param_list.append(intercept[..., jnp.newaxis])

        if log_concentration.ndim > beta.ndim - 1:
            param_list.append(jnp.squeeze(log_concentration, axis=-1)[..., jnp.newaxis])
        else:
            param_list.append(log_concentration[..., jnp.newaxis])

        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            if zero_logit.ndim > beta.ndim - 1:
                param_list.append(jnp.squeeze(zero_logit, axis=-1)[..., jnp.newaxis])
            else:
                param_list.append(zero_logit[..., jnp.newaxis])

        theta = jnp.concatenate(param_list, axis=-1)
        return theta

    def reconstruct_parameters(
        self, flat_params: jnp.ndarray, template: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct parameters from flattened array."""
        n_features = template["beta"].shape[-1]

        beta = flat_params[..., :n_features]
        intercept = flat_params[..., n_features]
        log_concentration = flat_params[..., n_features + 1]

        result = {
            "beta": beta,
            "intercept": intercept[..., jnp.newaxis],
            "log_concentration": log_concentration[..., jnp.newaxis],
        }

        if self.zero_inflated:
            zero_logit = flat_params[..., n_features + 2]
            result["zero_logit"] = zero_logit[..., jnp.newaxis]

        return result
