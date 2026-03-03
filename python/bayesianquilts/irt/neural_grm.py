import pathlib

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
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
    """Graded Response Model with a monotone neural network response function.

    Instead of the standard sigmoid link P(Y >= k) = sigma(a*(theta - b_k)),
    this model uses P(Y >= k) = g_i(a*(theta - b_k)) where g_i is a monotone
    neural network. When ``per_item_nn=True`` (default), each item gets its own
    mixture-of-sigmoids link function, allowing genuinely different ICC shapes
    per item. When ``per_item_nn=False``, a single shared g is used across all
    items.

    Monotonicity is guaranteed by using a mixture-of-sigmoids architecture:
        g_i(z) = Σ_h pi_{ih} × sigmoid(softplus(a_{ih}) × z + b_{ih})
    which is a weighted average of monotone increasing sigmoid functions.
    """

    response_type = "polytomous"
    nn_hidden_sizes: Any = nnx.data(None)
    per_item_nn: Any = nnx.data(None)

    def __init__(
        self,
        item_keys,
        num_people,
        nn_hidden_sizes=(4,),
        per_item_nn=True,
        num_groups=None,
        data=None,
        person_key="person",
        dim=1,
        decay=0.25,
        positive_discriminations=True,
        missing_val=-9999,
        full_rank=False,
        eta_scale=0.1,
        kappa_scale=0.5,
        weight_exponent=1.0,
        response_cardinality=5,
        discrimination_guess=None,
        include_independent=False,
        vi_mode='advi',
        imputation_model=None,
        dtype=jnp.float64,
    ):
        self.nn_hidden_sizes = nn_hidden_sizes
        self.per_item_nn = per_item_nn
        super(NeuralGRModel, self).__init__(
            item_keys=item_keys,
            num_people=num_people,
            num_groups=num_groups,
            data=data,
            person_key=person_key,
            dim=dim,
            decay=decay,
            positive_discriminations=positive_discriminations,
            missing_val=missing_val,
            full_rank=full_rank,
            eta_scale=eta_scale,
            kappa_scale=kappa_scale,
            weight_exponent=weight_exponent,
            response_cardinality=response_cardinality,
            discrimination_guess=discrimination_guess,
            include_independent=include_independent,
            vi_mode=vi_mode,
            imputation_model=imputation_model,
            dtype=dtype,
        )
        self.create_distributions()

    def _monotone_forward(self, z, nn_params):
        """Dispatch to shared or per-item monotone forward based on self.per_item_nn."""
        if self.per_item_nn:
            return self._per_item_monotone_forward(z, nn_params)
        return self._shared_monotone_forward(z, nn_params)

    def _shared_monotone_forward(self, z, nn_params):
        """Shared monotone response function: z (any shape) -> probabilities in (0, 1).

        Uses a **mixture-of-sigmoids** architecture with parameters shared across
        all items:

            g(z) = Σ_h  softmax(w_h) × sigmoid(softplus(a_h) × z + b_h)

        Parameters:
        - nn_w0: (batch..., H, 1) — component slopes (softplus → positive)
        - nn_b0: (batch..., H) — component offsets
        - nn_w1: (batch..., 1, H) — mixing logits (softmax → positive, sum to 1)

        Args:
            z: Array of any shape (batch..., spatial...).
            nn_params: Dict with keys nn_w0, nn_b0, nn_w1, nn_b1, ...

        Returns:
            Array of same shape as z, with values in (0, 1).
        """
        original_shape = z.shape

        # Determine batch dims from weight shape: w has (batch..., fan_out, fan_in)
        slopes_raw = nn_params['nn_w0']  # (batch..., H, 1)
        n_batch_dims = slopes_raw.ndim - 2
        batch_shape = z.shape[:n_batch_dims]
        spatial_shape = z.shape[n_batch_dims:]
        flat_size = 1
        for s in spatial_shape:
            flat_size *= s

        # Slopes: softplus to ensure positive (monotone increasing in z)
        slopes = jax.nn.softplus(slopes_raw)  # (batch..., H, 1)
        slopes = slopes[..., 0]  # (batch..., H)

        # Offsets: unconstrained shift for each sigmoid component
        offsets = nn_params['nn_b0']  # (batch..., H)

        # Mixing weights: softmax over components
        mix_logits = nn_params['nn_w1']  # (batch..., 1, H)
        mix_weights = jax.nn.softmax(mix_logits[..., 0, :], axis=-1)  # (batch..., H)

        # Flatten spatial dims of z: (batch..., M)
        z_flat = z.reshape(*batch_shape, flat_size)

        # Compute each component: sigmoid(slope_k * z + offset_k)
        # z_flat: (batch..., M), slopes: (batch..., H)
        # -> expand: (batch..., M, 1) * (batch..., 1, H) -> (batch..., M, H)
        components = jax.nn.sigmoid(
            z_flat[..., :, jnp.newaxis] * slopes[..., jnp.newaxis, :]
            + offsets[..., jnp.newaxis, :]
        )

        # Weighted average: (batch..., M, H) * (batch..., 1, H) -> sum -> (batch..., M)
        output = jnp.sum(
            components * mix_weights[..., jnp.newaxis, :], axis=-1
        )

        return output.reshape(original_shape)

    def _per_item_monotone_forward(self, z, nn_params):
        """Per-item monotone response function using item-specific mixture-of-sigmoids.

        Each item i has its own mixture:
            g_i(z) = Σ_h pi_{ih} × sigmoid(alpha_{ih} × z + beta_{ih})

        This allows different items to have genuinely different ICC shapes
        (asymmetric, variable steepness), producing predictions that a logistic
        GRM cannot replicate.

        Parameters have an item dimension (I) prepended:
        - nn_w0: (batch..., I, H, 1) — per-item slopes (softplus → positive)
        - nn_b0: (batch..., I, H) — per-item offsets
        - nn_w1: (batch..., I, 1, H) — per-item mixing logits

        Args:
            z: (..., N, D, I, K-1) — the scaled offsets for each person/item/threshold.
            nn_params: Dict with keys nn_w0, nn_b0, nn_w1, ...

        Returns:
            Array of same shape as z, with values in (0, 1).
        """
        # nn_w0: (batch..., I, H, 1)
        slopes_raw = nn_params['nn_w0']
        # batch dims = total ndim - 3 (I, H, 1)
        n_batch_dims = slopes_raw.ndim - 3
        I = slopes_raw.shape[n_batch_dims]

        # Slopes: softplus for positivity → (batch..., I, H)
        slopes = jax.nn.softplus(slopes_raw)[..., 0]  # (batch..., I, H)
        offsets = nn_params['nn_b0']  # (batch..., I, H)

        # Mixing weights: softmax over H → (batch..., I, H)
        mix_logits = nn_params['nn_w1']  # (batch..., I, 1, H)
        mix_weights = jax.nn.softmax(mix_logits[..., 0, :], axis=-1)  # (batch..., I, H)

        # z shape: (batch..., N, D, I, K-1)
        # We need to broadcast with per-item params along the I dimension.
        # Add H dim to z: (batch..., N, D, I, K-1, 1)
        z_expanded = z[..., jnp.newaxis]

        # Reshape slopes/offsets/mix to broadcast with z:
        # From (batch..., I, H) -> (batch..., 1, 1, I, 1, H)
        # Insert N and D dims (size 1) after batch dims, and K-1 dim (size 1) after I
        for _ in range(2):  # insert N, D dims
            slopes = jnp.expand_dims(slopes, axis=n_batch_dims)
            offsets = jnp.expand_dims(offsets, axis=n_batch_dims)
            mix_weights = jnp.expand_dims(mix_weights, axis=n_batch_dims)
        # Now (batch..., 1, 1, I, H) — insert K-1 dim before H
        slopes = jnp.expand_dims(slopes, axis=-2)       # (batch..., 1, 1, I, 1, H)
        offsets = jnp.expand_dims(offsets, axis=-2)      # (batch..., 1, 1, I, 1, H)
        mix_weights = jnp.expand_dims(mix_weights, axis=-2)  # (batch..., 1, 1, I, 1, H)

        # Compute sigmoid components: (batch..., N, D, I, K-1, H)
        components = jax.nn.sigmoid(z_expanded * slopes + offsets)

        # Weighted sum over H: (batch..., N, D, I, K-1)
        output = jnp.sum(components * mix_weights, axis=-1)

        return output

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

        # Apply monotone NN to -scaled (matching GRM's sigmoid(-scaled) convention)
        # so that P(Y >= k) increases with ability (theta) and decreases with threshold (b_k)
        cum_probs = self._monotone_forward(-scaled, nn_params)  # P(Y >= k)

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

        # --- NN weight priors (mixture-of-sigmoids) ---
        # When per_item_nn=True, each item gets its own mixture parameters with
        # an I dimension prepended:
        #   nn_w0: (I, H, 1)  slopes
        #   nn_b0: (I, H)     offsets
        #   nn_w1: (I, 1, H)  mixing logits
        #   nn_b1: (I, 1)     unused placeholder
        # When per_item_nn=False (shared), shapes are the same as before:
        #   nn_w0: (H, 1), nn_b0: (H,), nn_w1: (1, H), nn_b1: (1,)
        I = self.num_items
        sizes = list(self.nn_hidden_sizes)
        layer_sizes = [1] + sizes + [1]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            w_name = f'nn_w{i}'
            b_name = f'nn_b{i}'

            if self.per_item_nn:
                w_shape = (I, fan_out, fan_in)
                b_shape = (I, fan_out)
                w_batch_ndims = 3
                b_batch_ndims = 2
            else:
                w_shape = (fan_out, fan_in)
                b_shape = (fan_out,)
                w_batch_ndims = 2
                b_batch_ndims = 1

            grm_joint_distribution_dict[w_name] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(w_shape, dtype=self.dtype),
                    scale=jnp.ones(w_shape, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=w_batch_ndims,
            )
            # Use wider prior on offsets (b0) to allow diverse sigmoid shifts
            b_scale = 2.0 if i == 0 else 1.0
            grm_joint_distribution_dict[b_name] = tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros(b_shape, dtype=self.dtype),
                    scale=b_scale * jnp.ones(b_shape, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=b_batch_ndims,
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

    def save_to_disk(self, path):
        """Save NeuralGRModel to disk (YAML config + HDF5 params)."""
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            '_class_name': 'NeuralGRModel',
            'item_keys': list(self.item_keys),
            'num_people': int(self.num_people),
            'nn_hidden_sizes': list(self.nn_hidden_sizes),
            'dim': int(self.dimensions),
            'decay': float(self.dimensional_decay),
            'positive_discriminations': bool(self.positive_discriminations),
            'missing_val': int(self.missing_val),
            'full_rank': bool(self.full_rank),
            'eta_scale': float(self.eta_scale),
            'kappa_scale': float(self.kappa_scale.flatten()[0]) if hasattr(self.kappa_scale, 'flatten') else float(self.kappa_scale),
            'weight_exponent': float(self.weight_exponent),
            'response_cardinality': int(self.response_cardinality),
            'include_independent': bool(self.include_independent),
            'per_item_nn': bool(self.per_item_nn),
            'vi_mode': str(self.vi_mode),
            'dtype': 'float64' if self.dtype == jnp.float64 else 'float32',
        }
        with open(path / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        with h5py.File(path / 'params.h5', 'w') as f:
            if self.params is not None and hasattr(self.params, 'items'):
                grp = f.create_group('params')
                for k, v in self.params.items():
                    arr = np.array(v)
                    if arr.size > 0:
                        grp.create_dataset(k, data=arr)

    @classmethod
    def load_from_disk(cls, path):
        """Load NeuralGRModel from disk."""
        path = pathlib.Path(path)
        with open(path / 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        config.pop('_class_name', None)
        dtype_str = config.pop('dtype', 'float64')
        dtype = jnp.float64 if dtype_str == 'float64' else jnp.float32
        config['nn_hidden_sizes'] = tuple(config.get('nn_hidden_sizes', [32]))
        config['per_item_nn'] = config.get('per_item_nn', False)
        config['dtype'] = dtype

        instance = cls(**config)

        if (path / 'params.h5').exists():
            with h5py.File(path / 'params.h5', 'r') as f:
                if 'params' in f:
                    instance.params = {k: jnp.array(v) for k, v in f['params'].items()}

        return instance

    def simulate_data(self, abilities=None, seed=0):
        """Generate synthetic response data from the fitted model.

        Uses calibrated_expectations for NN weights and item parameters.
        Returns (N, I) integer response matrix with values in [0, K-1].

        Args:
            abilities: Optional ability array. If None, uses model's calibrated.
            seed: Random seed (int) for categorical sampling.
        """
        discrimination = self.calibrated_expectations['discriminations']
        if abilities is None:
            abilities = self.calibrated_expectations['abilities']

        nn_params = {
            k: self.calibrated_expectations[k]
            for k in self._get_nn_param_names()
        }

        difficulties0 = self.calibrated_expectations['difficulties0']
        if 'ddifficulties' in self.calibrated_expectations:
            ddifficulties = self.calibrated_expectations['ddifficulties']
        else:
            # K=2: no additional thresholds beyond difficulties0
            ddifficulties = jnp.zeros((*difficulties0.shape[:-1], 0))

        probs = self.neural_grm_model_prob_d(
            abilities,
            discrimination,
            difficulties0,
            ddifficulties,
            nn_params,
        )
        # Ensure valid probability simplex
        probs = jnp.clip(probs, 1e-10, None)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)

        response_rv = tfd.Categorical(probs=probs)
        responses = response_rv.sample(seed=jax.random.PRNGKey(seed))
        # Ensure integer responses in valid range
        responses = jnp.clip(responses, 0, self.response_cardinality - 1).astype(jnp.int32)
        return np.array(responses)
