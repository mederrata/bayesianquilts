import jax
import jax.numpy as jnp
import numpy as np
from typing import Any

from flax import nnx
from jax.scipy.special import xlogy
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.irt.irt import IRTModel


class NeuralGRModel(IRTModel):
    """Graded Response Model with a shared monotone neural network response function.

    Instead of the standard sigmoid link P(Y >= k) = sigma(a*(theta - b_k)),
    this model uses P(Y >= k) = g(a*(theta - b_k)) where g is a monotone
    neural network shared across all items. Item-specific discrimination (a_i)
    and difficulty thresholds (b_{i,k}) are retained.

    Monotonicity is guaranteed by:
    - Positive weights via Softplus applied inside the forward pass
    - Monotone activation (ReLU)
    - Final sigmoid output
    """

    response_type = "polytomous"
    nn_hidden_sizes: Any = nnx.data(None)

    def __init__(self, *args, nn_hidden_sizes=(32, 32), **kwargs):
        self.nn_hidden_sizes = nn_hidden_sizes
        super(NeuralGRModel, self).__init__(*args, **kwargs)
        self.create_distributions()

    def _monotone_forward(self, z, nn_params):
        """Shared monotone NN: z (any shape) -> probabilities in (0, 1).

        z = a_i * (theta_j - b_{i,k})  [scalar input per (person, item, threshold)]

        Weights are stored unconstrained; Softplus is applied here to ensure positivity.
        Forward: h = relu(softplus(W1) @ z + b1); ... ; out = sigmoid(softplus(Wn) @ h + bn)

        Args:
            z: Array of any shape (batch..., spatial...). The NN operates elementwise
                over the spatial dims. Batch dims must match the leading dims of the weights.
            nn_params: Dict with keys nn_w0, nn_b0, nn_w1, nn_b1, ..., nn_wL, nn_bL.
                Weights may have extra leading batch dims (e.g., sample dim from surrogate).

        Returns:
            Array of same shape as z, with values in (0, 1).
        """
        original_shape = z.shape

        # Determine batch dims from weight shape: w has (batch..., fan_out, fan_in)
        w0 = nn_params['nn_w0']
        n_batch_dims = w0.ndim - 2
        batch_shape = z.shape[:n_batch_dims]
        spatial_shape = z.shape[n_batch_dims:]
        flat_size = 1
        for s in spatial_shape:
            flat_size *= s

        # Flatten spatial dims: (batch..., M, 1) where M = product of spatial dims
        h = z.reshape(*batch_shape, flat_size, 1)

        n_layers = len(nn_params) // 2
        for i in range(n_layers):
            w = jax.nn.softplus(nn_params[f'nn_w{i}'])  # (batch..., fan_out, fan_in)
            b = nn_params[f'nn_b{i}']  # (batch..., fan_out)
            # h: (batch..., M, fan_in) @ w_T: (batch..., fan_in, fan_out)
            #  -> (batch..., M, fan_out)
            w_T = jnp.swapaxes(w, -1, -2)
            h = jnp.matmul(h, w_T) + b[..., jnp.newaxis, :]

            if i < n_layers - 1:
                h = jax.nn.relu(h)
            else:
                h = jax.nn.sigmoid(h)

        # h: (batch..., M, 1) -> reshape to original
        return h.reshape(original_shape)

    def _get_nn_param_names(self):
        """Return sorted list of NN parameter names."""
        sizes = list(self.nn_hidden_sizes)
        layer_sizes = [1] + sizes + [1]
        names = []
        for i in range(len(layer_sizes) - 1):
            names.extend([f'nn_w{i}', f'nn_b{i}'])
        return names

    def _extract_nn_params(self, params):
        """Extract NN parameters from a params dict."""
        return {k: params[k] for k in self._get_nn_param_names() if k in params}

    def neural_grm_model_prob(self, abilities, discriminations, difficulties, nn_params):
        """Compute P(Y=k) using the neural network response function.

        Args:
            abilities: (N, D, 1, 1) or (S, N, D, 1, 1)
            discriminations: (1, D, I, 1) or (S, 1, D, I, 1)
            difficulties: (1, D, I, K-1) or (S, 1, D, I, K-1) — cumulative thresholds
            nn_params: dict of NN weight/bias arrays

        Returns:
            probs: (S, N, I, K) or (N, I, K) — category probabilities
        """
        if self.include_independent:
            abilities = jnp.pad(
                abilities,
                [(0, 0)] * (len(discriminations.shape) - 3) + [(1, 0)] + [(0, 0)] * 2,
            )

        offsets = difficulties - abilities  # (..., N, D, I, K-1)
        scaled = offsets * discriminations  # (..., N, D, I, K-1)

        # Apply monotone NN instead of sigmoid
        cum_probs = self._monotone_forward(scaled, nn_params)  # P(Y >= k)

        # Pad: P(Y >= 0) = 1 on left, P(Y >= K) = 0 on right
        cum_probs = jnp.pad(
            cum_probs,
            ([(0, 0)] * (len(cum_probs.shape) - 1) + [(1, 0)]),
            mode="constant",
            constant_values=1,
        )
        cum_probs = jnp.pad(
            cum_probs,
            ([(0, 0)] * (len(cum_probs.shape) - 1) + [(0, 1)]),
            mode="constant",
            constant_values=0,
        )
        probs = cum_probs[..., :-1] - cum_probs[..., 1:]

        # Weight by discrimination and sum over dimensions axis
        weights = (
            jnp.abs(discriminations) ** self.weight_exponent
            / jnp.sum(jnp.abs(discriminations) ** self.weight_exponent, axis=-3)[
                ..., jnp.newaxis, :, :
            ]
        )
        probs = jnp.sum(probs * weights, axis=-3)
        return probs

    def neural_grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties, nn_params
    ):
        """Convenience: construct cumulative difficulties then call neural_grm_model_prob."""
        d0 = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(d0, axis=-1)
        return self.neural_grm_model_prob(
            abilities, discriminations, difficulties, nn_params
        )

    def predictive_distribution(
        self,
        data,
        discriminations,
        difficulties0,
        ddifficulties,
        abilities,
        **kwargs
    ):
        """Compute predictive distribution using the neural GRM.

        Same interface as GRModel.predictive_distribution but extracts NN params
        from kwargs and uses neural_grm_model_prob.
        """
        nn_params = self._extract_nn_params(kwargs)

        ddifficulties = jnp.where(
            ddifficulties < 1e-1, 1e-1 * jnp.ones_like(ddifficulties), ddifficulties
        )
        difficulties = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(difficulties, axis=-1)

        rank = len(abilities.shape)
        batch_shape = abilities.shape[: (rank - 4)]
        batch_ndims = len(batch_shape)

        people = data[self.person_key].astype(jnp.int32)
        choices = jnp.concat([data[i][:, jnp.newaxis] for i in self.item_keys], axis=-1)

        bad_choices = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)
        choices = jnp.where(bad_choices, jnp.zeros_like(choices), choices)

        for _ in range(batch_ndims):
            choices = choices[jnp.newaxis, ...]

        abilities = abilities[:, people, ...]

        response_probs = self.neural_grm_model_prob(
            abilities, discriminations, difficulties, nn_params
        )
        imputed_lp = jnp.sum(xlogy(response_probs, response_probs), axis=-1)

        rv_responses = tfd.Categorical(probs=response_probs)
        log_probs = rv_responses.log_prob(choices)

        imputation_pmfs = data.get('_imputation_pmfs')
        if imputation_pmfs is not None:
            log_rp = jnp.log(jnp.maximum(response_probs, 1e-30))
            log_q = jnp.log(jnp.maximum(imputation_pmfs, 1e-30))
            rb = jax.scipy.special.logsumexp(
                log_rp + log_q[jnp.newaxis, ...], axis=-1
            )
            log_probs = jnp.where(bad_choices[jnp.newaxis, ...], rb, log_probs)
        else:
            log_probs = jnp.where(
                bad_choices[jnp.newaxis, ...], imputed_lp, log_probs
            )

        log_probs = jnp.sum(log_probs, axis=-1)

        return {
            "log_likelihood": log_probs,
            "discriminations": discriminations,
            "rv": rv_responses,
        }

    def log_likelihood(
        self,
        data,
        discriminations,
        difficulties0,
        ddifficulties,
        abilities,
        *args,
        **kwargs
    ):
        prediction = self.predictive_distribution(
            data,
            discriminations,
            difficulties0,
            ddifficulties,
            abilities,
            *args,
            **kwargs
        )
        return prediction["log_likelihood"]

    def create_distributions(self, grouping_params=None):
        """Create prior and surrogate distributions.

        Same IRT priors as GRModel for abilities, mu, difficulties, discriminations, etc.
        Additional priors for NN weights: Normal(0, 1) unconstrained (Softplus applied
        inside _monotone_forward).
        """
        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }
        self.bijectors["eta"] = tfb.Softplus()
        self.bijectors["kappa"] = tfb.Softplus()
        self.bijectors["discriminations"] = tfb.Softplus()
        self.bijectors["ddifficulties"] = tfb.Softplus()
        self.bijectors["eta_a"] = tfb.Softplus()
        self.bijectors["kappa_a"] = tfb.Softplus()
        self.bijectors["xi"] = tfb.Softplus()

        K = self.response_cardinality

        # --- Standard IRT priors (same as GRModel) ---
        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=mu,
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
            discriminations=(
                (
                    lambda eta, kappa: tfd.Independent(
                        AbsHorseshoe(scale=eta * kappa), reinterpreted_batch_ndims=4
                    )
                )
                if self.positive_discriminations
                else (
                    lambda eta, xi, kappa: tfd.Independent(
                        tfd.Horseshoe(
                            loc=jnp.zeros(
                                (1, self.dimensions, self.num_items, 1),
                                dtype=self.dtype,
                            ),
                            scale=eta * kappa,
                        ),
                        reinterpreted_batch_ndims=4,
                    )
                )
            ),
            ddifficulties=tfd.Independent(
                tfd.HalfNormal(
                    scale=jnp.ones(
                        (
                            1,
                            self.dimensions,
                            self.num_items,
                            self.response_cardinality - 2,
                        ),
                        dtype=self.dtype,
                    )
                ),
                reinterpreted_batch_ndims=4,
            ),
            eta=tfd.Independent(
                tfd.HalfNormal(
                    scale=self.eta_scale
                    * jnp.ones((1, 1, self.num_items, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            ),
            kappa=lambda kappa_a: tfd.Independent(
                SqrtInverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    1.0 / kappa_a,
                ),
                reinterpreted_batch_ndims=4,
            ),
            kappa_a=tfd.Independent(
                tfd.InverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype)
                    / self.kappa_scale**2,
                ),
                reinterpreted_batch_ndims=4,
            ),
        )

        # --- Abilities prior (same as GRModel, no grouping support for now) ---
        if grouping_params is not None:
            raise NotImplementedError(
                "NeuralGRModel does not yet support grouping_params."
            )

        grm_joint_distribution_dict["abilities"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros(
                    (
                        self.num_people,
                        (
                            self.dimensions
                            if not self.include_independent
                            else self.dimensions - 1
                        ),
                        1,
                        1,
                    ),
                    dtype=self.dtype,
                ),
                scale=jnp.ones(
                    (
                        self.num_people,
                        (
                            self.dimensions
                            if not self.include_independent
                            else self.dimensions - 1
                        ),
                        1,
                        1,
                    ),
                    dtype=self.dtype,
                ),
            ),
            reinterpreted_batch_ndims=4,
        )

        # --- NN weight priors ---
        sizes = list(self.nn_hidden_sizes)
        layer_sizes = [1] + sizes + [1]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w_name = f'nn_w{i}'
            b_name = f'nn_b{i}'

            grm_joint_distribution_dict[w_name] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros((fan_out, fan_in), dtype=self.dtype),
                    scale=jnp.ones((fan_out, fan_in), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=2,
            )
            grm_joint_distribution_dict[b_name] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros((fan_out,), dtype=self.dtype),
                    scale=jnp.ones((fan_out,), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=1,
            )
            # Identity bijectors for NN params (Softplus applied inside forward pass)
            self.bijectors[w_name] = tfb.Identity()
            self.bijectors[b_name] = tfb.Identity()

        self.joint_prior_distribution = tfd.JointDistributionNamed(
            grm_joint_distribution_dict
        )
        self.prior_distribution = self.joint_prior_distribution
        self.var_list = list(self.joint_prior_distribution.model.keys())

        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.joint_prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
            )
        )
        self.params = self.surrogate_parameter_initializer()

    def unormalized_log_prob(self, data, prior_weight=1., **params):
        """Compute unnormalized log probability (prior + likelihood - entropy)."""
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        weights = prediction["discriminations"]
        weights = weights / jnp.sum(weights, axis=-3, keepdims=True)
        entropy = -xlogy(weights, weights) / params["eta"]
        entropy = jnp.sum(entropy, axis=[-1, -2, -3, -4])

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )
        min_val = jnp.min(finite_portion) - 1.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        return jnp.astype(prior_weight, log_prior.dtype) * (log_prior - entropy) + jnp.sum(
            log_likelihood, axis=-1
        )

    def simulate_data(self, abilities=None):
        """Generate synthetic response data from the fitted model.

        Uses calibrated_expectations for NN weights and item parameters.
        Returns (N, I) integer response matrix.
        """
        discrimination = self.calibrated_expectations['discriminations']
        if abilities is None:
            abilities = self.calibrated_expectations['abilities']

        nn_params = {
            k: self.calibrated_expectations[k]
            for k in self._get_nn_param_names()
        }

        probs = self.neural_grm_model_prob_d(
            abilities,
            discrimination,
            self.calibrated_expectations['difficulties0'],
            self.calibrated_expectations['ddifficulties'],
            nn_params,
        )
        response_rv = tfd.Categorical(probs=probs)
        responses = response_rv.sample(seed=jax.random.PRNGKey(0))
        return responses
