#!/usr/bin/env python3

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

from bayesianquilts.irt.irt import IRTModel


class FactorizedGRModel(IRTModel):
    """Factorized Graded Response Model for IRT.

    Items are grouped by scale indices, with each scale having its own
    discrimination, difficulty, and ability parameters.

    Supports optional stochastic imputation of missing responses via an
    imputation model.
    """

    response_type = "polytomous"

    def __init__(self, scale_indices, *args, discrimination_prior_scale=2.0, **kwargs):
        """Initialize model based on scale indices.

        Args:
            scale_indices (list(list(int))): Indices for the items per scale.
            discrimination_prior_scale (float): HalfNormal scale for the
                discrimination prior (default 2.0).
        """
        if isinstance(scale_indices, dict):
            self.scale_names = list(scale_indices.keys())
            self.scale_indices = list(scale_indices.values())
        else:
            self.scale_names = None
            self.scale_indices = scale_indices
        super(FactorizedGRModel, self).__init__(*args, **kwargs)
        self.discrimination_prior_scale = discrimination_prior_scale
        self.dimensions = len(scale_indices)
        self.create_distributions()

    def create_distributions(self):
        """Create scale-factorized prior and surrogate distributions."""
        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }

        self.bijectors["discriminations"] = tfb.Softplus()
        self.bijectors["ddifficulties"] = tfb.Softplus()

        grm_joint_distribution_dict = {}

        for j, indices in enumerate(self.scale_indices):
            self.bijectors[f"discriminations_{j}"] = tfb.Softplus()
            self.bijectors[f"ddifficulties_{j}"] = tfb.Softplus()
            self.bijectors[f"mu_{j}"] = tfb.Identity()
            grm_joint_distribution_dict = {
                **grm_joint_distribution_dict,
                **self.gen_discrim_prior(j, indices),
                **self.gen_difficulty_prior(j, indices),
                **self.gen_ability_prior(j),
            }

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

    def grm_model_prob(self, abilities, discriminations, difficulties):
        logits = difficulties - abilities  # N x D x I x K-1
        logits = logits * discriminations
        probs = jax.nn.sigmoid(-logits)
        probs = jnp.pad(
            probs,
            ([(0, 0)] * (len(probs.shape) - 1) + [(1, 0)]),
            mode="constant",
            constant_values=1,
        )
        probs = jnp.pad(
            probs,
            ([(0, 0)] * (len(probs.shape) - 1) + [(0, 1)]),
            mode="constant",
            constant_values=0,
        )
        probs = probs[..., :-1] - probs[..., 1:]
        return probs

    def grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties
    ):
        d0 = jnp.concat([difficulties0, ddifficulties], axis=-1)
        difficulties = jnp.cumsum(d0, axis=-1)
        return self.grm_model_prob(abilities, discriminations, difficulties)

    def fit_dim(self, batched_data_factory, *, dim: int, **kwargs):
        """Fit a single dimension as an independent univariate GRModel.

        Creates a 1D GRModel using only the items belonging to
        ``self.scale_indices[dim]``, fits it, and returns the fitted
        model along with losses and parameters.

        Args:
            batched_data_factory: Callable returning data iterator.
                Batches must contain all item keys (the method filters
                to the relevant subset).
            dim: Which dimension (index into ``self.scale_indices``) to fit.
            **kwargs: Forwarded to ``GRModel.fit`` (e.g. ``batch_size``,
                ``dataset_size``, ``num_epochs``, ``learning_rate``,
                ``imputation_model``, ``compute_elpd_loo``, etc.).

        Returns:
            Tuple of ``(univariate_model, losses, params)``.
        """
        from bayesianquilts.irt.grm import GRModel

        if dim >= self.dimensions:
            raise ValueError(
                f"dim={dim} out of range for model with "
                f"{self.dimensions} dimensions"
            )

        indices = self.scale_indices[dim]
        dim_item_keys = [self.item_keys[i] for i in indices]

        # Inherit imputation model from kwargs or from self
        imputation_model = kwargs.pop('imputation_model', self.imputation_model)

        uni_model = GRModel(
            item_keys=dim_item_keys,
            num_people=self.num_people,
            person_key=self.person_key,
            dim=1,
            response_cardinality=self.response_cardinality,
            eta_scale=self.eta_scale,
            positive_discriminations=self.positive_discriminations,
            discrimination_prior=getattr(self, 'discrimination_prior', 'half_normal'),
            discrimination_prior_scale=self.discrimination_prior_scale,
            dtype=self.dtype,
            imputation_model=imputation_model,
        )

        # The data factory may contain all items; GRModel.predictive_distribution
        # only reads its own item_keys, so no filtering is needed.
        res = uni_model.fit(batched_data_factory, **kwargs)
        losses, params = res[0], res[1]

        return uni_model, losses, params

    def assemble_from_dims(self, dim_models, n_samples=64, seed=42):
        """Reassemble fitted univariate GRModels into this FactorizedGRModel.

        Takes a dict ``{dim: GRModel}`` of fitted univariate models (one per
        dimension, as returned by ``fit_dim``) and populates this model's
        ``calibrated_expectations``, ``surrogate_sample``, and surrogate
        ``params`` from the per-dimension posteriors.

        Args:
            dim_models: Dict mapping dimension index to a fitted GRModel.
                Must have one entry per dimension (0..D-1).
            n_samples: Number of surrogate posterior samples to draw from
                each univariate model.
            seed: Random seed for sampling.

        Returns:
            self (for chaining).
        """
        import jax

        if set(dim_models.keys()) != set(range(self.dimensions)):
            raise ValueError(
                f"Expected models for dims {list(range(self.dimensions))}, "
                f"got {list(dim_models.keys())}"
            )

        # Sample from each univariate model's surrogate
        dim_samples = {}
        for d, uni_model in dim_models.items():
            surrogate = uni_model.surrogate_distribution_generator(uni_model.params)
            key = jax.random.PRNGKey(seed + d)
            dim_samples[d] = surrogate.sample(n_samples, seed=key)

        # Map into factorized param names
        assembled_samples = {}
        for d in range(self.dimensions):
            s = dim_samples[d]
            assembled_samples[f"discriminations_{d}"] = s["discriminations"]
            assembled_samples[f"mu_{d}"] = s["mu"]
            assembled_samples[f"difficulties0_{d}"] = s["difficulties0"]
            assembled_samples[f"ddifficulties_{d}"] = s["ddifficulties"]
            assembled_samples[f"abilities_{d}"] = s["abilities"]

        self.surrogate_sample = assembled_samples
        self.calibrated_expectations = {
            k: jnp.mean(v, axis=0) for k, v in assembled_samples.items()
        }
        self.calibrated_sd = {
            k: jnp.std(v, axis=0) for k, v in assembled_samples.items()
        }

        # Also update the FactorizedGRM surrogate params from the
        # shared variables (discriminations, difficulties, abilities).
        # The surrogate param keys use backslash-delimited paths.
        shared_vars = ["discriminations", "mu", "difficulties0", "ddifficulties", "abilities"]
        for d, uni_model in dim_models.items():
            for var in shared_vars:
                for suffix in list(self.params.keys()):
                    if not suffix.startswith(f"{var}_{d}\\"):
                        continue
                    # e.g. suffix = "discriminations_0\softplus\normal\loc"
                    # corresponding GRModel key = "discriminations\softplus\normal\loc"
                    grm_key = var + suffix[len(f"{var}_{d}"):]
                    if grm_key in uni_model.params:
                        self.params[suffix] = uni_model.params[grm_key]

        self._dim_models = dim_models
        return self

    def _find_surrogate_param(self, var_name: str, param_type: str) -> Optional[str]:
        """Find the surrogate parameter key for a given variable and param type.

        Searches ``self.params`` for keys matching the pattern
        ``<var_name>\\...\\<param_type>`` (e.g. ``abilities_0\\identity\\normal\\loc``).

        Returns the full key, or ``None`` if not found.
        """
        if not isinstance(self.params, dict):
            return None
        suffix = f"\\{param_type}"
        for key in self.params:
            parts = key.split("\\")
            if parts[0] == var_name and key.endswith(suffix):
                return key
        return None

    def standardize_abilities(self, weights=None, reference_idx=None):
        """Rescale per-scale parameters so abilities are N(0, 1) per dimension.

        Uses the surrogate distribution's loc/scale parameters directly
        to compute per-dimension mean and standard deviation.

        Operates on the scale-factorized parameter names
        (``abilities_j``, ``discriminations_j``, etc.).

        Args:
            weights: Optional (N,) per-person weights for weighted mean/std.
            reference_idx: Optional array of person indices whose abilities
                define the standardization statistics per dimension. All
                abilities are shifted and scaled using the reference subset's
                mean and std. Useful when a reference subpopulation (e.g.,
                the general population group) should anchor the scale.

        Returns:
            dict with ``mu`` and ``sigma`` arrays of shape (D,).
        """
        if not isinstance(self.params, dict):
            raise ValueError("No params — fit or load the model first")

        D = self.dimensions
        mu = jnp.zeros(D, dtype=self.dtype)
        sigma = jnp.ones(D, dtype=self.dtype)

        for d in range(D):
            loc_key = self._find_surrogate_param(f"abilities_{d}", "loc")
            scale_key = (
                self._find_surrogate_param(f"abilities_{d}", "scale")
                or self._find_surrogate_param(f"abilities_{d}", "log_scale")
            )
            if loc_key is None:
                continue

            ab_loc = self.params[loc_key]  # (N, 1, 1, 1)
            ab_1d = ab_loc[:, 0, 0, 0] if ab_loc.ndim >= 4 else ab_loc.reshape(-1)

            # Select reference subset
            if reference_idx is not None:
                ref_idx = jnp.asarray(reference_idx)
                ab_ref = ab_1d[ref_idx]
            else:
                ab_ref = ab_1d

            if weights is not None:
                w = jnp.asarray(weights, dtype=self.dtype)
                if reference_idx is not None:
                    w = w[ref_idx]
                w = w / jnp.sum(w)
                mu = mu.at[d].set(jnp.sum(w * ab_ref))
                sigma_val = jnp.sqrt(jnp.sum(w * (ab_ref - mu[d])**2))
                if scale_key is not None:
                    s = self.params[scale_key]
                    if "log_scale" in scale_key:
                        s = jnp.exp(s)
                    s_1d = s[:, 0, 0, 0] if s.ndim >= 4 else s.reshape(-1)
                    s_ref = s_1d[ref_idx] if reference_idx is not None else s_1d
                    sigma_val = jnp.sqrt(sigma_val**2 + jnp.sum(w * s_ref**2))
                sigma = sigma.at[d].set(sigma_val)
            else:
                mu = mu.at[d].set(jnp.mean(ab_ref))
                var_loc = jnp.var(ab_ref)
                if scale_key is not None:
                    s = self.params[scale_key]
                    if "log_scale" in scale_key:
                        s = jnp.exp(s)
                    s_1d = s[:, 0, 0, 0] if s.ndim >= 4 else s.reshape(-1)
                    s_ref = s_1d[ref_idx] if reference_idx is not None else s_1d
                    var_scale = jnp.mean(s_ref.reshape(-1)**2)
                    sigma = sigma.at[d].set(jnp.sqrt(var_loc + var_scale))
                else:
                    sigma = sigma.at[d].set(jnp.sqrt(var_loc))

        sigma = jnp.where(sigma < 1e-8, 1.0, sigma)

        # Apply rescaling to surrogate distribution parameters (loc/scale)
        # in unconstrained space, accounting for bijectors.
        #
        # Constrained-space transforms (all scalar-affine):
        #   abilities:       a  -> (a - mu) / sigma
        #   discriminations: disc -> disc * sigma
        #   difficulties0:   d0 -> (d0 - mu) / sigma
        #   ddifficulties:   dd -> dd / sigma
        #
        # For each variable we:
        #   1. Map loc to constrained space via bijector.forward
        #   2. Apply the affine transform
        #   3. Map back via bijector.inverse
        #   4. Adjust scale using the Jacobian ratio at old/new loc
        for d in range(D):
            mu_d = mu[d]
            sigma_d = sigma[d]

            var_transforms = [
                (f"abilities_{d}",       lambda x, m=mu_d, s=sigma_d: (x - m) / s),
                (f"discriminations_{d}", lambda x, m=mu_d, s=sigma_d: x * s),
                (f"mu_{d}",             lambda x, m=mu_d, s=sigma_d: (x - m) / s),
                (f"difficulties0_{d}",   lambda x, m=mu_d, s=sigma_d: (x - m) / s),
                (f"ddifficulties_{d}",   lambda x, m=mu_d, s=sigma_d: x / s),
            ]

            for var_name, constrained_fn in var_transforms:
                loc_key = self._find_surrogate_param(var_name, "loc")
                if loc_key is None:
                    continue

                bijector = self.bijectors.get(var_name, tfb.Identity())

                # Transform loc
                old_loc = self.params[loc_key]
                old_constrained = bijector.forward(old_loc)
                new_constrained = constrained_fn(old_constrained)
                new_loc = bijector.inverse(new_constrained)
                self.params[loc_key] = new_loc

                # Transform scale: in unconstrained space, the scale changes by
                # the ratio of the inverse-bijector Jacobians at new vs old loc,
                # times the constrained-space scale factor.
                # Since all transforms are scalar-affine, the constrained scale
                # factor is constant (1/sigma, sigma, etc.).
                for scale_type in ("scale", "log_scale"):
                    scale_key = self._find_surrogate_param(var_name, scale_type)
                    if scale_key is None:
                        continue

                    # Jacobian of bijector.forward at the unconstrained loc
                    # For identity: 1. For softplus: sigmoid(x).
                    fwd_log_jac_old = bijector.forward_log_det_jacobian(old_loc)
                    fwd_log_jac_new = bijector.forward_log_det_jacobian(new_loc)

                    # unconstrained_scale_new / unconstrained_scale_old =
                    #   (constrained_scale_ratio) * exp(fwd_log_jac_old) / exp(fwd_log_jac_new)
                    # because d(unconstrained)/d(constrained) = 1/fwd_jac
                    #
                    # The constrained scale ratio for scalar-affine f(x)=ax+b is |a|.
                    # We compute it numerically for generality.
                    eps = jnp.where(
                        jnp.abs(old_constrained) > 1e-6,
                        1e-6 * jnp.abs(old_constrained),
                        1e-6 * jnp.ones_like(old_constrained),
                    )
                    constrained_ratio = jnp.abs(
                        (constrained_fn(old_constrained + eps) - new_constrained) / eps
                    )

                    log_scale_ratio = (
                        jnp.log(constrained_ratio + 1e-30)
                        + fwd_log_jac_old - fwd_log_jac_new
                    )

                    if scale_type == "log_scale":
                        self.params[scale_key] = self.params[scale_key] + log_scale_ratio
                    else:
                        self.params[scale_key] = self.params[scale_key] * jnp.exp(log_scale_ratio)

        # Invalidate cached samples/expectations since params changed
        self.surrogate_sample = None
        self.calibrated_expectations = None
        self.calibrated_sd = None

        return {'mu': mu, 'sigma': sigma}

    def transform(self, params):
        """Reassemble scale-factorized parameters into full tensors."""
        discriminations = []
        d0 = []
        dd = []
        for j, indices in enumerate(self.scale_indices):
            update = jnp.transpose(params[f"discriminations_{j}"], [3, 0, 1, 2, 4])[
                :, :, 0, 0, 0
            ]
            update_d0 = jnp.transpose(params[f"difficulties0_{j}"], [3, 0, 1, 2, 4])[
                :, :, 0, 0, 0
            ]
            update_dd = jnp.transpose(params[f"ddifficulties_{j}"], [4, 3, 0, 1, 2])[
                :, :, :, 0, 0
            ]
            S = update.shape[1]
            output_array = jnp.zeros((S, self.num_items), dtype=update.dtype)
            output_array_d0 = jnp.zeros((S, self.num_items), dtype=update.dtype)
            output_array_dd = jnp.zeros(
                (S, self.num_items, self.response_cardinality - 2), dtype=update.dtype
            )

            discriminations += [
                output_array.at[:, indices].set(update.T)[..., jnp.newaxis]
            ]
            d0 += [output_array_d0.at[:, indices].set(update_d0.T)[..., jnp.newaxis]]
            dd += [output_array_dd.at[:, indices].set(update_dd.T)[..., jnp.newaxis]]
        discriminations = jnp.concat(discriminations, axis=-1)
        abilities = jnp.concat(
            [params[f"abilities_{j}"] for j in range(self.dimensions)], axis=-3
        )
        d0 = jnp.concat(d0, axis=-1)
        dd = jnp.concat(dd, axis=-1)
        _shape = discriminations.shape
        _rank = len(_shape)
        dd = jnp.transpose(
            dd, [t for t in range(_rank - 2)] + [_rank, _rank - 2, _rank - 1]
        )
        discriminations = jnp.transpose(
            discriminations, [t for t in range(_rank - 2)] + [_rank - 1, _rank - 2]
        )
        discriminations = discriminations[..., jnp.newaxis, :, :, jnp.newaxis]

        d0 = jnp.transpose(d0, [t for t in range(_rank - 2)] + [_rank - 1, _rank - 2])
        d0 = d0[..., jnp.newaxis, :, :, jnp.newaxis]
        dd = dd[..., jnp.newaxis, :, :, :]
        params["discriminations"] = discriminations
        params["difficulties0"] = d0
        params["ddifficulties"] = dd
        diff = jnp.concat([d0, dd], axis=-1)
        diff = jnp.cumsum(diff, axis=-1)
        params["difficulties"] = diff
        params["abilities"] = abilities

        return params

    def _mcmc_log_prior(self, item_params):
        """Log prior with per-scale ``mu_j`` integrated out analytically.

        In ``gen_difficulty_prior`` we have:
            mu_{j} ~ Normal(d0_loc, 5)
            difficulties0_{j} | mu_{j} ~ Normal(mu_{j}, 1)
        which gives the marginal
            difficulties0_{j} ~ Normal(d0_loc, sqrt(26)).

        Priors on ``discriminations_{j}`` and ``ddifficulties_{j}`` are
        unchanged (they don't depend on mu).
        """
        K = self.response_cardinality
        d0_loc = -(K - 2) / 2.0
        marginal_scale = jnp.sqrt(jnp.asarray(26.0, dtype=self.dtype))

        lp = jnp.asarray(0.0, dtype=self.dtype)
        model = self.joint_prior_distribution.model
        for j, _indices in enumerate(self.scale_indices):
            d0_name = f"difficulties0_{j}"
            if d0_name in item_params:
                diff0 = jnp.asarray(item_params[d0_name], dtype=self.dtype)
                d0_prior = tfd.Independent(
                    tfd.Normal(
                        loc=jnp.full(diff0.shape, d0_loc, dtype=self.dtype),
                        scale=marginal_scale
                        * jnp.ones(diff0.shape, dtype=self.dtype),
                    ),
                    reinterpreted_batch_ndims=diff0.ndim,
                )
                lp = lp + d0_prior.log_prob(diff0)

        # Reuse unchanged priors for every other MCMC-sampled var
        # (discriminations_j, ddifficulties_j, ...).
        skip_prefixes = ("mu_", "difficulties0_", "abilities", "abilities_")
        for name, factor in model.items():
            if name.startswith(skip_prefixes) or name not in item_params:
                continue
            val = jnp.asarray(item_params[name], dtype=self.dtype)
            if callable(factor):
                import inspect
                sig = inspect.signature(factor)
                parent_kwargs = {}
                skip_this = False
                for pname in sig.parameters:
                    if pname in item_params:
                        parent_kwargs[pname] = item_params[pname]
                    else:
                        skip_this = True
                        break
                if skip_this:
                    continue
                dist = factor(**parent_kwargs)
            else:
                dist = factor
            lp = lp + dist.log_prob(val)
        return lp

    def gen_discrim_prior(self, j, indices):
        out = {}
        out[f"discriminations_{j}"] = tfd.Independent(
            tfd.HalfNormal(
                scale=self.discrimination_prior_scale * tf.ones((1, 1, len(indices), 1))
            ),
            reinterpreted_batch_ndims=4,
        )
        return out

    def gen_ability_prior(self, j):
        out = {}
        out[f"abilities_{j}"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.zeros((self.num_people, 1, 1, 1), self.dtype),
                scale=5 * jnp.ones((self.num_people, 1, 1, 1), self.dtype),
            ),
            reinterpreted_batch_ndims=4,
        )
        return out

    def gen_difficulty_prior(self, j, indices):
        # Hierarchical prior: mu_{j} ~ N(-(K-2)/2, 5) sets the per-scale
        # difficulty center, with difficulties0_{j} | mu_{j} ~ N(mu_{j}, 1).
        # The mu prior is centered at -(K-2)/2 so the median threshold
        # starts near 0 after cumsum with ddifficulties ~ HalfNormal(1).
        K = self.response_cardinality
        d0_loc = -(K - 2) / 2.0
        n_items = len(indices)
        dtype = self.dtype
        out = {}
        out[f"mu_{j}"] = tfd.Independent(
            tfd.Normal(
                loc=jnp.full((1, 1, n_items, 1), d0_loc, dtype=dtype),
                scale=5.0 * jnp.ones((1, 1, n_items, 1), dtype=dtype),
            ),
            reinterpreted_batch_ndims=4,
        )
        # JointDistributionNamed resolves parents by inspecting argument
        # names, so the conditional must have a parameter named mu_{j}.
        # Build the function dynamically so the signature matches.
        _globs = {"tfd": tfd, "jnp": jnp, "_n": n_items, "_d": dtype}
        _code = (
            f"def _d0_prior(mu_{j}):\n"
            f"    return tfd.Independent(\n"
            f"        tfd.Normal(\n"
            f"            loc=jnp.asarray(mu_{j}, dtype=_d),\n"
            f"            scale=jnp.ones((1, 1, _n, 1), dtype=_d),\n"
            f"        ),\n"
            f"        reinterpreted_batch_ndims=4,\n"
            f"    )\n"
        )
        _ns = {}
        exec(_code, _globs, _ns)  # noqa: S102 - dynamic signature for JointDistributionNamed
        out[f"difficulties0_{j}"] = _ns["_d0_prior"]
        out[f"ddifficulties_{j}"] = tfd.Independent(
            tfd.HalfNormal(
                scale=tf.ones(
                    (
                        1,
                        1,
                        len(indices),
                        self.response_cardinality - 2,
                    ),
                    dtype=self.dtype,
                ),
            ),
            reinterpreted_batch_ndims=4,
        )
        return out

    def _response_probs_grid(self, theta_grid, **item_params):
        """Compute response probs on a theta grid for the factorized model.

        For D dimensions, uses 1D quadrature with discrimination-weighted
        averaging across dimensions. Computes per-dimension GRM probs
        directly without calling transform().
        """
        Q = len(theta_grid)
        I = self.num_items
        K = self.response_cardinality

        # Accumulate per-dimension probs and disc weights
        all_probs = jnp.zeros((Q, I, K))
        all_disc = jnp.zeros((I,))

        for d, indices in enumerate(self.scale_indices):
            idx = jnp.array(indices)
            disc_d = item_params[f"discriminations_{d}"].squeeze()
            diff0_d = item_params[f"difficulties0_{d}"].squeeze()
            ddiff_key = f"ddifficulties_{d}"
            if ddiff_key in item_params:
                ddiff_d = item_params[ddiff_key]
                # Squeeze all but the last dim for multi-category diffs
                while ddiff_d.ndim > 2:
                    ddiff_d = ddiff_d.squeeze(0)
                if ddiff_d.ndim == 1:
                    ddiff_d = ddiff_d[..., jnp.newaxis]
                ddiff_safe = jax.nn.softplus(ddiff_d)
                difficulties_d = jnp.concat(
                    [diff0_d[..., jnp.newaxis], ddiff_safe], axis=-1
                )
            else:
                difficulties_d = diff0_d[..., jnp.newaxis]
            difficulties_d = jnp.cumsum(difficulties_d, axis=-1)

            # theta_grid: (Q,) → (Q, 1, 1), disc: (n_d,) → (1, n_d, 1)
            # difficulties: (n_d, K-1) → (1, n_d, K-1)
            theta_col = theta_grid[:, None, None]
            disc_col = jnp.abs(disc_d)[None, :, None]
            diff_col = difficulties_d[None, :, :]

            logits = diff_col - theta_col  # (Q, n_d, K-1)
            logits = logits * disc_col
            cum_probs = jax.nn.sigmoid(-logits)
            cum_probs = jnp.pad(
                cum_probs,
                [(0, 0), (0, 0), (1, 0)],
                constant_values=1.0,
            )
            cum_probs = jnp.pad(
                cum_probs,
                [(0, 0), (0, 0), (0, 1)],
                constant_values=0.0,
            )
            probs_d = cum_probs[..., :-1] - cum_probs[..., 1:]  # (Q, n_d, K)
            probs_d = jnp.clip(probs_d, 1e-30, None)

            # Scatter into full item array
            mean_disc = jnp.mean(jnp.abs(disc_d))
            all_probs = all_probs.at[:, idx, :].set(probs_d * mean_disc)
            all_disc = all_disc.at[idx].set(mean_disc)

        # Normalize by total discrimination weight
        all_probs = all_probs / jnp.maximum(all_disc[None, :, None], 1e-30)
        return all_probs  # (Q, I, K)

    def predictive_distribution(
        self, data, discriminations, difficulties0, ddifficulties, abilities, **kwargs
    ):
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

        bad_choices = (
            (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)
        )

        for _ in range(batch_ndims):
            choices = choices[jnp.newaxis, ...]

        abilities = abilities[..., people, :, :, :]

        response_probs = self.grm_model_prob(abilities, discriminations, difficulties)
        discrimination_weights = jnp.abs(discriminations) / jnp.sum(
            jnp.abs(discriminations), axis=-3, keepdims=True
        )

        response_probs = jnp.sum(
            response_probs * discrimination_weights, axis=-3
        )

        rv_responses = tfd.Categorical(probs=response_probs)

        log_probs = rv_responses.log_prob(choices)

        imputation_pmfs = data.get('_imputation_pmfs')
        if imputation_pmfs is not None:
            # Analytic Rao-Blackwellization: log[ sum_k q(k) * p(Y=k|phi) ]
            log_rp = jnp.log(jnp.maximum(response_probs, 1e-30))  # (S, N, I, K)
            log_q = jnp.log(jnp.maximum(imputation_pmfs, 1e-30))  # (N, I, K)
            rb = jax.scipy.special.logsumexp(
                log_rp + log_q[jnp.newaxis, ...], axis=-1
            )  # (S, N, I)
            log_probs = jnp.where(bad_choices[jnp.newaxis, ...], rb, log_probs)
        else:
            log_probs = jnp.where(
                bad_choices[jnp.newaxis, ...], jnp.zeros_like(log_probs), log_probs
            )

        log_probs = jnp.sum(log_probs, axis=-1)

        return {
            "log_likelihood": log_probs,
            "discriminations": discriminations,
            "rv": rv_responses,
        }

    def unormalized_log_prob(self, data, prior_weight=1.0, **params):
        _params = params.copy()
        if 'difficulties' not in _params:
            _params = self.transform(_params)
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **_params)
        log_likelihood = prediction["log_likelihood"]

        finite_portion = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.zeros_like(log_likelihood),
        )

        min_val = jnp.min(finite_portion) - 5.0
        log_likelihood = jnp.where(
            jnp.isfinite(log_likelihood),
            log_likelihood,
            jnp.ones_like(log_likelihood) * min_val,
        )
        if 'sample_weights' in data:
            sw = jnp.asarray(data['sample_weights'], dtype=log_likelihood.dtype)
            weighted_ll = jnp.sum(sw[None, :] * log_likelihood, axis=-1)
        else:
            weighted_ll = jnp.sum(log_likelihood, axis=-1)
        return prior_weight * log_prior + weighted_ll
