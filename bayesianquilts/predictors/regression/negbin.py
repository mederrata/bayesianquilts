import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from typing import Dict, Any
from bayesianquilts.model import BayesianModel
from bayesianquilts.metrics.ais import AutoDiffLikelihoodMixin


class NegativeBinomialRegression(BayesianModel):
    def __init__(
        self,
        input_dim,
        zero_inflated=False,
        dtype=jnp.float64,
        prior_scale_beta=10.0,
        prior_scale_intercept=10.0,
        prior_scale_log_concentration=2.0,
        prior_scale_zero_logit=2.0,
        **kwargs,
    ):
        """
        Negative Binomial GLM with optional zero-inflation.

        Args:
            input_dim: Number of input features
            zero_inflated: If True, uses zero-inflated negative binomial
            dtype: Data type for computations
            prior_scale_beta: Prior scale for beta
            prior_scale_intercept: Prior scale for intercept
            prior_scale_log_concentration: Prior scale for log_concentration
            prior_scale_zero_logit: Prior scale for zero_logit

        Parameters:
            beta: Regression coefficients (input_dim,)
            intercept: Intercept term (1,)
            log_concentration: Log of concentration parameter (1,)
            zero_logit: Logit of zero-inflation probability (1,) [only if zero_inflated=True]
        """
        super().__init__(dtype=dtype, **kwargs)
        self.input_dim = input_dim
        self.zero_inflated = zero_inflated
        self.prior_scale_beta = prior_scale_beta
        self.prior_scale_intercept = prior_scale_intercept
        self.prior_scale_log_concentration = prior_scale_log_concentration
        self.prior_scale_zero_logit = prior_scale_zero_logit

        if zero_inflated:
            self.var_list = ["beta", "intercept", "log_concentration", "zero_logit"]
        else:
            self.var_list = ["beta", "intercept", "log_concentration"]

        self.create_distributions()

    def create_distributions(self):
        # Surrogate: Mean-field Normal
        dist_dict = {
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
            "log_concentration": tfd.Independent(
                tfd.Normal(
                    jnp.zeros(1, dtype=self.dtype),
                    jnp.array(self.prior_scale_log_concentration, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=1,
            ),
        }

        if self.zero_inflated:
            dist_dict["zero_logit"] = tfd.Independent(
                tfd.Normal(
                    jnp.zeros(1, dtype=self.dtype),
                    jnp.array(self.prior_scale_zero_logit, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=1,
            )

        self.prior_distribution = tfd.JointDistributionNamed(dist_dict)

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
            # intercept is (S,) so we need (S, 1) for broadcasting
            eta = jnp.einsum("nd,sd->sn", X, beta) + intercept[:, jnp.newaxis]
        elif beta.ndim == 3:
            # Case 3: AIS per-datum parameters (S, N, n_feat)
            # X (N, D), beta (S, N, D) -> (S, N)
            if X.ndim == 2:
                # Standard case: X is (N, D)
                eta = jnp.einsum("nd,snd->sn", X, beta)
            else:
                # Subset case: X is (B, D), beta is still (S, B, D)
                eta = jnp.einsum("bd,sbd->sb", X, beta)

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
        # When mean is (S, N) and concentration is (S,), we need (S, 1)
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

        return mean, total_count_exp, probs

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        X = data["X"].astype(self.dtype)
        y = data["y"].astype(self.dtype)

        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]

        has_sample_dim = beta.ndim > 1

        # Priors
        log_prior = self.prior_distribution.log_prob(params)

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

        return log_lik + log_prior * prior_weight

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

            # Broadcast zero_prob if needed (S,) -> (S, 1) or (S, N)
            if zero_prob.ndim < nb_logprob_zero.ndim:
                 zero_prob_bc = zero_prob[..., jnp.newaxis]
            else:
                 zero_prob_bc = zero_prob

            eps = jnp.array(1e-10, dtype=self.dtype)
            log_zero_prob = jnp.log(zero_prob_bc + eps)
            log_one_minus_zero_prob = jnp.log(
                jnp.array(1.0, dtype=self.dtype) - zero_prob_bc + eps
            )

            ll_zero = jnp.logaddexp(
                log_zero_prob, log_one_minus_zero_prob + nb_logprob_zero
            )
            ll_nonzero = log_one_minus_zero_prob + nb_logprob

            return jnp.where(y == 0, ll_zero, ll_nonzero)
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
        """Compute log-likelihood for Negative Binomial regression.
        
        Handles flattening of structured MCMC parameters (e.g. chains, samples)
        to ensure compatibility with AIS which expects (S, ...) batching.
        """
        # We need to flatten parameters if they correspond to multiple chains
        # so consistent with extract_parameters and AIS expectations of (S, N).
        
        flat_params = {}
        target_len = -1
        
        # Heuristic: check beta shape. If 3D (C, S, F) or more, flatten leading dims.
        # But allow (S, N, F) for specific AIS local-param usage.
        # We use data size N to disambiguate if possible, but params are usually (C, S, ...).
        
        # Safer strategy: flatten everything to match what extract_parameters does
        # extract_parameters broadcasts then concatenates.
        # Here we just want to execute log_likelihood.
        
        # If parameters are shaped (C, S, ...), we want to reshape to (C*S, ...)
        # Only if dimensions indicate extra structure.
        
        beta = params["beta"]
        if beta.ndim > 2:
             # Could be (C, S, F) or (S, N, F)
             # If middle dim is NOT data size (approximately), assume chain
             # But here we don't have N easily accessible without lookups
             pass

        # For AIS with this class, we assume standard global parameters unless
        # explicitly doing local variational etc. 
        # The safest fix for the reported issue (4, 2000, 4) is simply to flatten
        # any >2 dim beta to 2 dim, and >1 dim intercept to 1 dim.
        
        flat_params = params.copy()
        
        if "beta" in params:
            b = params["beta"]
            if b.ndim > 2:
                flat_params["beta"] = b.reshape(-1, b.shape[-1])
        
        for k in ["intercept", "log_concentration", "zero_logit"]:
            if k in params:
                v = params[k]
                if v.ndim > 1:
                    flat_params[k] = v.flatten()

        return self.model.log_likelihood(data, **flat_params)

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """Extract parameters into flattened array."""
        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]
        
        # Dynamic broadcasting to common batch shape
        batch_shapes = [beta.shape[:-1], intercept.shape, log_concentration.shape]
        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            batch_shapes.append(zero_logit.shape)
        target_shape = jnp.broadcast_shapes(*batch_shapes)

        beta = jnp.broadcast_to(beta, target_shape + (beta.shape[-1],))
        intercept = jnp.broadcast_to(intercept, target_shape)
        log_concentration = jnp.broadcast_to(log_concentration, target_shape)

        param_list = [beta, intercept[..., jnp.newaxis], log_concentration[..., jnp.newaxis]]
        
        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            zero_logit = jnp.broadcast_to(zero_logit, target_shape)
            param_list.append(zero_logit[..., jnp.newaxis])

        theta = jnp.concatenate(param_list, axis=-1)
        
        # Flatten structure (C, S, K) -> (C*S, K) for AIS vectorization
        if theta.ndim > 2:
            theta = theta.reshape(-1, theta.shape[-1])
            
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
            "intercept": intercept,
            "log_concentration": log_concentration,
        }

        if self.zero_inflated:
            zero_logit = flat_params[..., n_features + 2]
            result["zero_logit"] = zero_logit

        return result

    def log_likelihood_gradient(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> jnp.ndarray:
        """Compute gradient of log-likelihood w.r.t. parameters.

        Uses analytical gradients for standard Negative Binomial,
        falls back to AutoDiffLikelihoodMixin for Zero-Inflated case.
        """
        if self.zero_inflated:
            return super().log_likelihood_gradient(data, params)

        # Explicit Gradient for Negative Binomial
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        # Flatten params same as log_likelihood handling
        flat_params = params.copy()
        if "beta" in params:
             b = params["beta"]
             if b.ndim > 2:
                 flat_params["beta"] = b.reshape(-1, b.shape[-1])
        
        for k in ["intercept", "log_concentration"]:
            if k in params:
                v = params[k]
                if v.ndim > 1:
                    flat_params[k] = v.flatten()

        beta = flat_params["beta"]
        intercept = flat_params["intercept"]
        log_concentration = flat_params["log_concentration"]

        # Compute mu and other terms
        # Expect beta (S, D), intercept (S,), log_conc (S,)
        # X (N, D)
        
        # mu = exp(X @ beta + intercept)
        # Using eps for stability
        eps = 1e-6
        
        # Consistent broadcasting
        # eta: (S, N)
        if beta.ndim == 2:
            eta = jnp.dot(beta, X.T) + intercept[:, jnp.newaxis]
        else:
             # Should be caught by flattening, but handle (D,) case just in case
             eta = jnp.dot(X, beta) + intercept
             # This would be (N,), needs (S, N) broadcast if S=1
        
        mu = jnp.exp(eta)
        r = jnp.exp(log_concentration)
        if r.ndim == 1:
             r = r[:, jnp.newaxis] # (S, 1)

        # Gradients
        
        # dL/d(eta) = y - (y+r) * mu / (mu + r)
        #           = y - (y+r) * p_fail? 
        #           mu / (mu + r) = 1 - p_success
        
        term = (y[jnp.newaxis, :] + r) * mu / (mu + r)
        dl_deta = y[jnp.newaxis, :] - term  # (S, N)
        
        # dL/dbeta = dl_deta @ X
        # (S, N) @ (N, D) -> (S, D)
        # Wait, result needs to be (S, N, D) for AIS
        # AIS expects gradient per data point!
        
        # dL_n / dbeta = dl_deta_n * X_n
        # (S, N, 1) * (1, N, D) -> (S, N, D)
        dl_dbeta = dl_deta[..., jnp.newaxis] * X[jnp.newaxis, ...]
        
        # dL/dintercept
        dl_dintercept = dl_deta[..., jnp.newaxis] # (S, N, 1)
        
        # dL/dr
        # dL/dr = psi(y+r) - psi(r) + 1 + log(r/(mu+r)) - (r+y)/(mu+r)
        
        psi_y_r = jax.lax.digamma(y[jnp.newaxis, :] + r)
        psi_r = jax.lax.digamma(r)
        
        dl_dr = (psi_y_r - psi_r + 1.0 + jnp.log(r / (mu + r)) 
                 - (r + y[jnp.newaxis, :]) / (mu + r))
                 
        # dL/d(log_conc) = dL/dr * r
        dl_dlog_conc = dl_dr * r
        dl_dlog_conc = dl_dlog_conc[..., jnp.newaxis] # (S, N, 1)

        # Concatenate: beta, intercept, log_concentration
        # output shape: (S, N, K)
        
        gradients = jnp.concatenate([dl_dbeta, dl_dintercept, dl_dlog_conc], axis=-1)
        
        # Flatten sample dimension if it was flattened from chains
        # extract_parameters produces (S*C, K) so our (S, N, K) is correct assuming S includes chains
        
        return gradients

