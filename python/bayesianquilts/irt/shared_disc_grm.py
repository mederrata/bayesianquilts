"""Shared-discrimination GRM (Rasch-style) for ordinal IRT data.

A single discrimination is shared across all items in a scale; thresholds
remain per-item. Useful under heavy structural missingness where the full
GRM cannot reliably estimate per-item slopes.

Implementation: subclass `GRModel` and after the parent's full
`create_distributions` (which sets up abilities, mu, difficulties0,
discriminations, ddifficulties, surrogate, etc.), override just the
discriminations marginal to be shape (1, D, 1, 1) shared across items.
Re-build the surrogate so its discriminations component has the matching
shape.
"""
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd

from bayesianquilts.irt.grm import GRModel
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator


class SharedDiscGRModel(GRModel):
    """GRM with a single global discrimination shared across all items.

    Parameter shapes:
      * discriminations: (1, D, 1, 1)         # shared across items
      * difficulties0:   (1, D, I, 1)         # per-item base threshold
      * ddifficulties:   (1, D, I, K-2)       # per-item threshold increments
      * mu:              (1, D, I, 1)         # per-item difficulty location prior
      * abilities:       (N, D, 1, 1)         # per-person ability
    """

    response_type = "polytomous"

    def create_distributions(self, grouping_params=None):
        # Build the full GRModel prior first (abilities, mu, difficulties0,
        # discriminations, ddifficulties, optional horseshoe pieces, ...).
        super().create_distributions(grouping_params=grouping_params)

        prior_scale = getattr(self, 'discrimination_prior_scale', None)
        prior_type = getattr(self, 'discrimination_prior', 'half_normal')
        _scale = float(prior_scale) if prior_scale is not None else 2.0

        # Replace discriminations with a shared (1, D, 1, 1) version.
        model_dict = dict(self.joint_prior_distribution.model)
        disc_scale = jnp.asarray(
            _scale * jnp.ones(
                (1, self.dimensions, 1, 1), dtype=self.dtype),
            dtype=self.dtype,
        )
        if prior_type == 'half_cauchy':
            model_dict["discriminations"] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros_like(disc_scale),
                    scale=disc_scale,
                ),
                reinterpreted_batch_ndims=4,
            )
        else:
            # Default: HalfNormal. Horseshoe variants reuse HalfNormal here
            # because the per-item local shrinkage no longer applies.
            model_dict["discriminations"] = tfd.Independent(
                tfd.HalfNormal(scale=disc_scale),
                reinterpreted_batch_ndims=4,
            )

        self.joint_prior_distribution = tfd.JointDistributionNamed(model_dict)
        self.prior_distribution = self.joint_prior_distribution
        self.var_list = list(self.joint_prior_distribution.model.keys())

        # Rebuild surrogate so its discriminations component matches the
        # new (1, D, 1, 1) shape.
        self.surrogate_distribution_generator, self.surrogate_parameter_initializer = (
            build_factored_surrogate_posterior_generator(
                self.joint_prior_distribution,
                bijectors=self.bijectors,
                dtype=self.dtype,
                parameterization=self.parameterization,
                rank=self.rank,
            )
        )
        self.params = self.surrogate_parameter_initializer()
