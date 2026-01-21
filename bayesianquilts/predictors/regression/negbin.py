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
            # intercept may be (S,) or (S, 1) - squeeze to 1D then expand
            intercept_1d = jnp.squeeze(intercept) if intercept.ndim > 1 else intercept
            eta = jnp.einsum("nd,sd->sn", X, beta) + intercept_1d[:, jnp.newaxis]
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

        # Handle log_concentration that might be (S,), (S, 1), or (S, N, 1)
        if log_concentration.ndim == 3:
            concentration = jnp.exp(jnp.squeeze(log_concentration, axis=-1))
        elif log_concentration.ndim == 2 and log_concentration.shape[-1] == 1:
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

        probs = mean / (total_count_exp + mean)
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
            # Align zero_prob shape with nb_logprob_zero (Reference Shape)
            # zero_prob can be (S,), (S, 1), (S, N), (S, N, 1) etc.
            # nb_logprob_zero is typically (S, N)
            
            zero_prob_bc = zero_prob
            if zero_prob.ndim < nb_logprob_zero.ndim:
                 # Expand, e.g. (S,) -> (S, 1)
                 zero_prob_bc = zero_prob[..., jnp.newaxis]
            elif zero_prob.ndim > nb_logprob_zero.ndim:
                 # Squeeze, e.g. (S, N, 1) -> (S, N)
                 zero_prob_bc = jnp.squeeze(zero_prob, axis=-1)
            
            # Check if dimensions match now
            if zero_prob_bc.ndim != nb_logprob_zero.ndim:
                # Still failing? Try standard broadcast
                pass

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

    def tail_probability_mass(self, data, **params):
        X = data["X"].astype(self.dtype)
        y = data["y"].astype(self.dtype)
        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]

        mean, total_count, probs = self._compute_mean_and_concentration(
            X, beta, intercept, log_concentration
        )

        nb_dist = tfd.NegativeBinomial(total_count=total_count, probs=probs)
        # S(y) = P(Y >= y) = P(Y > y - 1)
        nb_survival = nb_dist.survival_function(y - 1)

        if self.zero_inflated:
            zero_logit = params["zero_logit"]
            zero_prob = jax.nn.sigmoid(zero_logit).astype(self.dtype)

            # Broadcast zero_prob to match nb_survival shape
            zero_prob_bc = zero_prob
            if zero_prob.ndim < nb_survival.ndim:
                 zero_prob_bc = zero_prob[..., jnp.newaxis]
            elif zero_prob.ndim > nb_survival.ndim:
                 zero_prob_bc = jnp.squeeze(zero_prob, axis=-1)

            # If y = 0, P(Y >= 0) = 1
            # If y > 0, P(Y >= y) = (1 - pi) * P_NB(Y >= y)
            return jnp.where(y == 0, 1.0, (1.0 - zero_prob_bc) * nb_survival)
        else:
            return nb_survival


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
        
        # UPDATE: AIS PyTree refactor uses (S, N, D) for local parameters.
        # Flattening breaks this structure.
        # We trust the model to handle the shapes passed.
        
        # Note: If legacy code passes (C, S, D), it might break if model doesn't support 3 dims.
        # But we prioritize correctness for (S, N) AIS.
        
        return self.model.log_likelihood(data, **params)

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
            # Explicit Gradient for Zero-Inflated Negative Binomial
            X = jnp.asarray(data["X"], dtype=self.dtype)
            y = jnp.asarray(data["y"], dtype=self.dtype) # (N,)

            beta = params["beta"]
            intercept = params["intercept"]
            log_concentration = params["log_concentration"]
            zero_logit = params["zero_logit"]

            # Shapes: beta (S, N, D) or (S, 1, D). X (N, D).
            # others: (S, N, 1) or (S, 1, 1) or (S, N) or (S, 1).
            
            # Expand X to (1, N, D)
            X = jnp.asarray(data["X"], dtype=self.dtype)
            X_expanded = X[jnp.newaxis, :, :] # (1, N, D)

            # Ensure params have atomic "N" dimension as broadcasting dim (axis -2 or -1 depending on rank)
            # beta (S, D) -> (S, 1, D)
            if beta.ndim == 2:
                beta_in = beta[:, jnp.newaxis, :]
            else:
                beta_in = beta # assume (S, N, D) or (S, 1, D)

            # intercept (S,) -> (S, 1)
            # (S, N) -> (S, N)
            def expand_scalar(p):
                 if p.ndim == 1:
                     return p[:, jnp.newaxis]
                 return p
            
            intercept_in = expand_scalar(intercept)
            
            # eta calculation
            # (S, 1, D) * (1, N, D) -> (S, N, D)
            # sum(-1) -> (S, N)
            eta = jnp.sum(beta_in * X_expanded, axis=-1) + intercept_in
            
            mu = jnp.exp(eta) # (S, N)

            # log_concentration (S,) -> (S, 1) -> broadcasting to (S, N) happens implicitly?
            # r = exp(log_conc). r is (S, 1).
            r = jnp.exp(expand_scalar(log_concentration))
            
            # zero_logit
            z_in = expand_scalar(zero_logit)
            pi = jax.nn.sigmoid(z_in) # (S, 1)

            
            # NB probability of 0: p0 = (r / (mu + r))^r
            # For numerical stability with large r, use log space
            # log_p0 = r * (log(r) - log(mu + r))
            log_p0 = r * (jnp.log(r) - jnp.log(mu + r))
            p0 = jnp.exp(log_p0) # (S, N)

            # Denominator for y=0 case: D = pi + (1 - pi) * p0
            # D = pi + p0 - pi*p0
            D = pi + (1.0 - pi) * p0

            # --- Gradients ---
            
            # Common terms
            # NB gradient term for mean: d(log P_NB)/d(eta)
            # For y > 0: y - (y+r) * mu / (mu+r)
            # For y = 0: - r * mu / (mu+r)
            
            grad_nb_eta_common = - (r * mu) / (mu + r) # This is the y=0 case of standard NB grad

            # 1. Gradient w.r.t. zero_logit
            # d(pi)/d(z) = pi * (1 - pi)
            
            # For y = 0: dL/dz = (1/D) * (1 - p0) * d(pi)/dz
            # = (1/D) * (1 - p0) * pi * (1 - pi)
            dl_dz_zero = (1.0 - p0) * pi * (1.0 - pi) / D
            
            # For y > 0: dL/dz = -pi (derived from d/dz log(1-pi))
            dl_dz_nonzero = -pi
            
            # Broadcast to (S, N)
            # dl_dz_nonzero is (S, 1), need (S, N) if y is (N,)
            dl_dz_nonzero = jnp.broadcast_to(dl_dz_nonzero, mu.shape)

            dl_dzero_logit = jnp.where(y == 0, dl_dz_zero, dl_dz_nonzero)

            # 2. Gradient w.r.t. eta
            # For y = 0: dL/d(eta) = ((1 - pi)/D) * d(P_NB(0))/d(eta)
            # d(P_NB(0))/d(eta) = p0 * grad_nb_eta_common
            # So: ((1 - pi) * p0 / D) * grad_nb_eta_common
            factor_zero = (1.0 - pi) * p0 / D
            dl_deta_zero = factor_zero * grad_nb_eta_common

            # For y > 0: Standard NB gradient
            # y - (y+r) * mu / (mu + r)
            term_nonzero = (y + r) * mu / (mu + r)
            dl_deta_nonzero = y - term_nonzero

            dl_deta = jnp.where(y == 0, dl_deta_zero, dl_deta_nonzero)

            # 3. Gradient w.r.t. log_concentration
            # dL/d(log_r) = dL/dr * r
            
            # Standard NB d/dr part (without the r mult)
            # For y > 0: this is complex digamma terms
            psi_y_r = jax.lax.digamma(y + r)
            psi_r = jax.lax.digamma(r)
            grad_nb_r_nonzero = (psi_y_r - psi_r + jnp.log(r / (mu + r)) 
                                 + 1.0 - (r + y) / (mu + r))
            
            # For y = 0: d(P_NB(0))/dr * (1/P_NB(0)) = log(r/(mu+r)) + mu/(mu+r)
            # Check: d/dr [ r log(r/(mu+r)) ] = log(...) + r * (1/r - 1/(mu+r)) = log + 1 - r/(mu+r) = log + mu/(mu+r)
            grad_nb_r_zero = jnp.log(r / (mu + r)) + mu / (mu + r)

            # Combine for full dL/d(log_r)
            # For y=0: factor_zero * d(P_NB(0))/d(log_r) ??? 
            # No, dL/d(log_r) = ((1-pi)/D) * d(P_NB(0))/dr * r
            # = factor_zero * grad_nb_r_zero * r
            dl_dlog_conc_zero = factor_zero * grad_nb_r_zero * r

            # For y > 0: grad_nb_r_nonzero * r
            dl_dlog_conc_nonzero = grad_nb_r_nonzero * r

            dl_dlog_conc = jnp.where(y == 0, dl_dlog_conc_zero, dl_dlog_conc_nonzero)


            # --- Assemble Gradients ---
            
            # dL/dbeta = dl_deta * X
            dl_dbeta = dl_deta[..., jnp.newaxis] * X[jnp.newaxis, ...] # (S, N, D)

            # dL/dintercept = dl_deta
            dl_dintercept = dl_deta[..., jnp.newaxis] # (S, N, 1)

            # dL/dlog_conc
            dl_dlog_conc = dl_dlog_conc[..., jnp.newaxis] # (S, N, 1)

            # dL/dzero_logit
            dl_dzero_logit = dl_dzero_logit[..., jnp.newaxis] # (S, N, 1)

            return {
                "beta": dl_dbeta,
                "intercept": dl_dintercept,
                "log_concentration": dl_dlog_conc,
                "zero_logit": dl_dzero_logit
            }

    def log_likelihood_hessian_diag(
        self, data: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute diagonal of Hessian of log-likelihood w.r.t. parameters.
        
        Uses analytical derivation for beta/intercept (via d^2L/deta^2) to avoid O(D^2) cost.
        Uses AD for scalar parameters (log_concentration, zero_logit).
        """
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype) # (N,)
        y_expanded = y[jnp.newaxis, :, jnp.newaxis] # (1, N, 1)

        # Cast parameters to model dtype to avoid TFP mismatch
        beta = params["beta"].astype(self.dtype)
        intercept = params["intercept"].astype(self.dtype)
        log_concentration = params["log_concentration"].astype(self.dtype)
        
        # --- 1. Compute d^2L / deta^2 analytically ---
        
        # Broadcasting setup (similar to gradient)
        X_expanded = X[jnp.newaxis, :, :] # (1, N, D)
        
        if beta.ndim == 2:
            beta_in = beta[:, jnp.newaxis, :]
        else:
            beta_in = beta

        def expand_scalar(p, ndim=2):
             """Expand scalar parameter to target ndim for broadcasting."""
             if p.ndim == 1:
                 if ndim == 2:
                     return p[:, jnp.newaxis]
                 elif ndim == 3:
                     return p[:, jnp.newaxis, jnp.newaxis]
             elif p.ndim == 2 and ndim == 3:
                 return p[:, :, jnp.newaxis]
             return p

        intercept_in = expand_scalar(intercept, ndim=3)
        eta = jnp.sum(beta_in * X_expanded, axis=-1, keepdims=True) + intercept_in
        mu = jnp.exp(eta) # (S, N, 1)

        r = jnp.exp(expand_scalar(log_concentration, ndim=3))
        
        # Determine d2_l_d_eta2
        # Common terms
        p = mu / (mu + r) # prob of success (NB) or mean-related
        
        # Non-zero case: d^2L / deta^2 = - (y+r) * p * (1-p)
        d2_l_d_eta2_nonzero = - (y_expanded + r) * p * (1.0 - p)
        
        
        if self.zero_inflated:
            zero_logit = params["zero_logit"].astype(self.dtype)
            z_in = expand_scalar(zero_logit, ndim=3)
            pi = jax.nn.sigmoid(z_in)
            
            log_p0 = r * (jnp.log(r) - jnp.log(mu + r))
            p0 = jnp.exp(log_p0)
            
            D = pi + (1.0 - pi) * p0
            w_0 = (1.0 - pi) * p0 / D
            
            # Zero case: d^2L / deta^2 = - r * p * w_0 * [ (1-p) - r * p * (1-w_0) ]
            d2_l_d_eta2_zero = - r * p * w_0 * ( (1.0 - p) - r * p * (1.0 - w_0) )
            
            d2_l_d_eta2 = jnp.where(y_expanded == 0, d2_l_d_eta2_zero, d2_l_d_eta2_nonzero)
            
            # --- 2. AD for scalars (log_concentration, zero_logit) ---
            # We define a helper that computes point-wise log-likelihood for one observation
            # and differentiate it twice w.r.t specific params.
            # This is slow if looped, but they are scalars so VMAP is efficient.
            
            # Or better, just implement analytical?
            # Analytical for log_r and zero_logit is messy. 
            # Let's use `jax.grad(jax.grad)` but vmapped over (S, N).
            
            # We need d^2 L / d (log_r)^2 and d^2 L / d z^2.
            
            # Helper to compute ELL for scalar inputs (per point)
            def scalar_ll(my_y, my_eta, my_log_r, my_z):
                # Similar to compiled log_likelihood logic but for scalars
                my_mu = jnp.exp(my_eta)
                my_r = jnp.exp(my_log_r)
                
                my_pi = jax.nn.sigmoid(my_z)
                
                # ZINB
                nb_lp = tfd.NegativeBinomial(total_count=my_r, probs=my_mu/(my_mu+my_r)).log_prob(my_y)
                # Note: TFP probs is p = mu / (mu+r) ? 
                # TFP probs argument is "probability of success". 
                # Mean is total_count * probs / (1-probs).
                # mu = r * p / (1-p) => mu/r = p/(1-p) => p = (mu/r) / (1 + mu/r) = mu / (mu+r).
                # Yes.
                
                nb_lp_zero = tfd.NegativeBinomial(total_count=my_r, probs=my_mu/(my_mu+my_r)).log_prob(0.)
                
                # Manual ZINB mixture
                eps = 1e-10
                log_zero = jnp.log(my_pi + eps)
                log_nonzero_weight = jnp.log(1.0 - my_pi + eps)
                
                ll_0 = jnp.logaddexp(log_zero, log_nonzero_weight + nb_lp_zero)
                ll_nz = log_nonzero_weight + nb_lp
                
                return jnp.where(my_y == 0, ll_0, ll_nz)

            # We want d2/d(log_r)2 and d2/d(z)2.
            # Inputs: y (scalar), eta (scalar), log_r (scalar), z (scalar)
            
            d2_logr_fn = jax.grad(jax.grad(scalar_ll, argnums=2), argnums=2)
            d2_z_fn = jax.grad(jax.grad(scalar_ll, argnums=3), argnums=3)
            
            # VMAP application
            # y (N,), eta (S, N), log_conc (S, 1), z (S, 1)
            # Broadcast scalars to (S, N)
            
            log_c_in = expand_scalar(log_concentration)
            if log_c_in.shape[1] == 1:
                log_c_in = jnp.tile(log_c_in, (1, y.shape[0]))
                
            z_in_full = expand_scalar(zero_logit)
            if z_in_full.shape[1] == 1:
                z_in_full = jnp.tile(z_in_full, (1, y.shape[0]))

            # We need to broadcast y to (S, N)
            y_in = jnp.tile(y[jnp.newaxis, :], (eta.shape[0], 1))
            
            # Squeeze eta from (S, N, 1) to (S, N) for vmap
            eta_flat = jnp.squeeze(eta, axis=-1)
            # log_c_in and z_in_full are already (S, N) after tiling
            log_c_flat = log_c_in
            z_flat = z_in_full
            
            d2_log_concentration = jax.vmap(jax.vmap(d2_logr_fn))(y_in, eta_flat, log_c_flat, z_flat)
            d2_zero_logit = jax.vmap(jax.vmap(d2_z_fn))(y_in, eta_flat, log_c_flat, z_flat)

        else:
             # Standard NB
             d2_l_d_eta2 = d2_l_d_eta2_nonzero
             
             # Need d2/d(log_r)2 for standard NB
             # Reuse helper?
             def scalar_ll_nb(my_y, my_eta, my_log_r):
                my_mu = jnp.exp(my_eta)
                my_r = jnp.exp(my_log_r)
                # TFP parameterization
                return tfd.NegativeBinomial(total_count=my_r, probs=my_mu/(my_mu+my_r)).log_prob(my_y)
                
             d2_logr_fn = jax.grad(jax.grad(scalar_ll_nb, argnums=2), argnums=2)
             
             log_c_in = expand_scalar(log_concentration)
             if log_c_in.shape[1] == 1:
                log_c_in = jnp.tile(log_c_in, (1, y.shape[0]))
             y_in = jnp.tile(y[jnp.newaxis, :], (eta.shape[0], 1))
             
             eta_flat = jnp.squeeze(eta, axis=-1)
             # log_c_in is already (S, N) after tiling
             log_c_flat = log_c_in

             d2_log_concentration = jax.vmap(jax.vmap(d2_logr_fn))(y_in, eta_flat, log_c_flat)
             d2_zero_logit = jnp.zeros_like(d2_log_concentration) # Dummy or not returned?
             
        
        # --- Assemble Diagonal Hessian ---
        
        # beta: d^2L / d beta_j^2 = (d^2L / d eta^2) * x_j^2
        # d2_l_d_eta2: (S, N, 1)
        # X: (N, D) -> X^2: (1, N, D)
        # Result: (S, N, D)
        
        # Expand X to (1, N, D) explicitly
        X2 = (X**2)[jnp.newaxis, :, :]
        d2_beta = d2_l_d_eta2 * X2
        
        # intercept: d^2L / d intercept^2 = d^2L / d eta^2
        d2_intercept = d2_l_d_eta2 # (S, N, 1)
        
        # log_conc
        d2_log_conc = d2_log_concentration[..., jnp.newaxis] # (S, N, 1)
        
        results = {
            "beta": d2_beta,
            "intercept": d2_intercept,
            "log_concentration": d2_log_conc,
        }
        
        if self.zero_inflated:
             results["zero_logit"] = d2_zero_logit[..., jnp.newaxis]
             
        return results

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        # Deprecated by AIS refactor, but kept if needed or remove?
        # The base class raises NotImplementedError. 
        # But we previously implemented it.
        # If we remove it, we strictly enforce PyTree AIS.
        # Let's keep it for now but maybe warn? Or just remove to align with plan.
        pass

        # Explicit Gradient for Negative Binomial
        X = jnp.asarray(data["X"], dtype=self.dtype)
        y = jnp.asarray(data["y"], dtype=self.dtype)

        # Access parameters directly to save RAM (avoid dict copy)
        # We assume params are already in the correct shape/type or references
        # If reshaping is needed, we do it on the specific array reference 
        
        beta = params["beta"]
        intercept = params["intercept"]
        log_concentration = params["log_concentration"]
        
        # Determine shapes
        # beta: (S, D) or (C, S, D) -> treat multiple batch dims as one effective batch S_eff
        # We need effective S to broadcast against N
        
        orig_beta_shape = beta.shape
        if beta.ndim > 2:
            # Flatten batch dimensions effectively for calculation
            # View, not copy if possible
            beta = beta.reshape(-1, orig_beta_shape[-1])
        
        if intercept.ndim > 1:
            intercept = intercept.flatten()
            
        if log_concentration.ndim > 1:
            log_concentration = log_concentration.flatten()
            
        # Standardize parameter dimensions: (S, D), (S,), (S,)
        
        # Compute eta: (S, N)
        # X: (N, D)
        if beta.ndim == 2:
            eta = jnp.dot(beta, X.T) + intercept[:, jnp.newaxis]
        else:
            # (D,) case
            eta = jnp.dot(X, beta) + intercept
            
        mu = jnp.exp(eta)
        r = jnp.exp(log_concentration)
        
        # Ensure r is (S, 1) for broadcasting against (S, N)
        if r.ndim == 1:
             r = r[:, jnp.newaxis]
        elif r.ndim == 0:
             # scalar case
             r = r
             
        # Gradients
        # dL/d(eta) = y - (y+r) * mu / (mu + r)
        # (S, N) broadcasting
        term = (y + r) * mu / (mu + r)
        dl_deta = y - term  # (S, N)
        
        # dL_n / dbeta = dl_deta_n * X_n
        # Expand dims for outer product per data point: (S, N, 1) * (1, N, D) -> (S, N, D)
        dl_dbeta = dl_deta[..., jnp.newaxis] * X[jnp.newaxis, ...]
        
        # dL/dintercept = dl_deta (expanded to match rank)
        dl_dintercept = dl_deta[..., jnp.newaxis] # (S, N, 1)
        
        # dL/dr
        psi_y_r = jax.lax.digamma(y + r)
        psi_r = jax.lax.digamma(r)
        
        dl_dr = (psi_y_r - psi_r + 1.0 + jnp.log(r / (mu + r)) 
                 - (r + y) / (mu + r))
                 
        # dL/d(log_conc) = dL/dr * r
        dl_dlog_conc = dl_dr * r
        dl_dlog_conc = dl_dlog_conc[..., jnp.newaxis] # (S, N, 1)
        
        # Concatenate: beta, intercept, log_concentration along last dim
        gradients = jnp.concatenate([dl_dbeta, dl_dintercept, dl_dlog_conc], axis=-1)
        
        # If input had extra batch dims, reshape gradients to match
        # Expected output for AIS is usually (S, N, K)
        # But if input was (C, S, ...), output might ideally be (C, S, N, K)
        # Current AIS implementation usually flattens to (S_eff, N, K) anyway.
        # We return the flattened batch version (S_eff, N, K) consistent with previous behavior.
        
        return gradients

