"""GDR2 (Generalized Decomposition prior on R²) for Bayesian shrinkage.

Implements the R2D2 and GDR2 priors from:
  Zhang, Nott, Gunawan, Maguire (2025) "Generalized Decomposition Priors on R²"
  Bayesian Analysis, advance publication. doi:10.1214/25-BA1524

The prior specifies:
  R² ~ Beta(a, b)                            # proportion of explained variance
  phi ~ Dirichlet(alpha) or LogisticNormal   # variance allocation across K groups
  omega² = R² / (1 - R²)                    # signal-to-noise ratio
  sigma_k = sqrt(phi_k * omega²)            # per-group scale
  beta_k ~ N(0, sigma_k² * I)               # coefficients

For the GDR2 variant, phi is drawn from a LogisticNormal distribution
(multivariate normal mapped through softmax), which allows flexible
covariance structure among the variance shares.
"""

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb


class GDR2Prior:
    """Generic GDR2/R2D2 prior for use with Decomposed parameter structures.

    This class computes:
      1. log p(R²) under a Beta prior
      2. log p(phi) under Dirichlet or LogisticNormal
      3. log p(coefficients | R², phi) under the induced Normal scales

    It can be used standalone or integrated into a JointDistributionNamed model.

    Args:
        n_components: Number of coefficient groups (K).
        component_sizes: List of ints — number of scalar parameters in each group.
            Used to normalize variance: group k gets scale sqrt(phi_k * omega² / n_k).
        mu_R2: Prior mean of R² (default 0.5).
        phi_R2: Prior precision of R² Beta distribution (default 2.0).
            Beta params are a = mu_R2 * phi_R2, b = (1 - mu_R2) * phi_R2.
        concentration: Base Dirichlet concentration parameter (default 1.0).
            For R2D2, alpha_k = concentration * weight_k.
        component_weights: Optional array of relative weights for Dirichlet alpha.
            Default: all ones (symmetric). Use dim_decay_factor**order for ANOVA.
        prior_type: "r2d2" for Dirichlet phi, "gdr2" for LogisticNormal phi.
        dtype: JAX dtype for computations.
    """

    def __init__(
        self,
        n_components,
        component_sizes,
        mu_R2=0.5,
        phi_R2=2.0,
        concentration=1.0,
        component_weights=None,
        prior_type="gdr2",
        dtype=jnp.float64,
    ):
        self.n_components = n_components
        self.component_sizes = np.array(component_sizes)
        self.mu_R2 = mu_R2
        self.phi_R2 = phi_R2
        self.concentration = concentration
        self.prior_type = prior_type
        self.dtype = dtype

        if component_weights is None:
            component_weights = np.ones(n_components)
        self.component_weights = np.array(component_weights, dtype=np.float64)

        # Dirichlet concentration vector
        self.alpha = self.concentration * self.component_weights
        self.alpha = jnp.array(self.alpha, dtype=dtype)

        # Beta distribution for R²
        a = mu_R2 * phi_R2
        b = (1.0 - mu_R2) * phi_R2
        self.R2_dist = tfd.Beta(
            concentration1=jnp.array(a, dtype=dtype),
            concentration0=jnp.array(b, dtype=dtype),
        )

        # Simplex distribution for phi
        if prior_type == "r2d2":
            self.phi_dist = tfd.Dirichlet(concentration=self.alpha)
        elif prior_type == "gdr2":
            # LogisticNormal: Normal in R^{K-1} mapped through SoftmaxCentered
            # Set the LN mean/scale to approximate Dirichlet(alpha)
            # via KL-minimization (see Zhang et al. §3.2)
            mu_ln, sigma_ln = self._dirichlet_to_logistic_normal(self.alpha)
            self._mu_ln = mu_ln
            self._sigma_ln = sigma_ln
            base_normal = tfd.MultivariateNormalDiag(
                loc=mu_ln, scale_diag=sigma_ln,
            )
            self.phi_dist = tfd.TransformedDistribution(
                distribution=base_normal,
                bijector=tfb.SoftmaxCentered(),
            )
        else:
            raise ValueError(f"Unknown prior_type: {prior_type}")

    @staticmethod
    def _dirichlet_to_logistic_normal(alpha):
        """Approximate Dirichlet(alpha) with LogisticNormal(mu, diag(sigma²)).

        Uses the moment-matching approach: match E[log(phi_k/phi_K)] and
        Var[log(phi_k/phi_K)] of the Dirichlet to the LN parameters.

        For Dirichlet(alpha), the log-ratio statistics are:
          E[log(phi_k/phi_K)] = psi(alpha_k) - psi(alpha_K)
          Var[log(phi_k/phi_K)] = trigamma(alpha_k) + trigamma(alpha_K)
        where psi is the digamma function and K is the last component.
        """
        K = len(alpha)
        # Digamma and trigamma
        psi = jnp.vectorize(lambda x: jax_digamma(x))
        psi_vals = _digamma(alpha)
        trigamma_vals = _trigamma(alpha)

        # Log-ratio moments: log(phi_k / phi_K) for k = 0, ..., K-2
        mu = psi_vals[:-1] - psi_vals[-1]
        sigma = jnp.sqrt(trigamma_vals[:-1] + trigamma_vals[-1])
        return mu, sigma

    def compute_scales(self, R2, phi):
        """Compute per-group scales from R² and phi.

        Args:
            R2: Scalar or (S,) array — explained variance fraction.
            phi: (K,) or (S, K) array — variance allocation on simplex.

        Returns:
            scales: (K,) or (S, K) array — per-group standard deviations.
        """
        R2 = jnp.clip(R2, 1e-6, 1.0 - 1e-6)
        omega2 = R2 / (1.0 - R2)

        # Per-group variance: phi_k * omega² / n_k
        n_k = jnp.array(self.component_sizes, dtype=self.dtype)
        if phi.ndim == 1:
            var_k = phi * omega2 / n_k
        else:
            # (S, K)
            if jnp.ndim(omega2) == 0:
                var_k = phi * omega2 / n_k[jnp.newaxis, :]
            else:
                var_k = phi * omega2[:, jnp.newaxis] / n_k[jnp.newaxis, :]

        return jnp.sqrt(jnp.maximum(var_k, 1e-20))

    def log_prob_R2(self, R2):
        """Log-probability of R² under the Beta prior."""
        return self.R2_dist.log_prob(R2)

    def log_prob_phi(self, phi):
        """Log-probability of phi under the simplex prior."""
        return self.phi_dist.log_prob(phi)

    def log_prob_coefficients(self, coefficients, scales):
        """Log-probability of coefficients given per-group scales.

        Args:
            coefficients: List of K arrays, each with shape (n_k_elements,)
                or (S, n_k_elements...) for batched evaluation.
            scales: (K,) or (S, K) — per-group scales from compute_scales().

        Returns:
            Scalar or (S,) log-probability.
        """
        lp = jnp.zeros((), dtype=self.dtype)
        for k, coef in enumerate(coefficients):
            if scales.ndim == 1:
                s = scales[k]
            else:
                s = scales[:, k]
                # Expand for broadcasting with coef shape
                while s.ndim < coef.ndim:
                    s = s[..., jnp.newaxis]
            # Sum over all non-batch dims to get scalar or (S,)
            element_lp = tfd.Normal(loc=0.0, scale=s).log_prob(coef)
            if coef.ndim > 1:
                lp = lp + jnp.sum(element_lp, axis=tuple(range(1, coef.ndim)))
            else:
                lp = lp + jnp.sum(element_lp)
        return lp

    def log_prob(self, R2, phi, coefficients):
        """Full GDR2 prior log-probability.

        Args:
            R2: Scalar or (S,) — explained variance fraction.
            phi: (K,) or (S, K) — variance allocation simplex.
            coefficients: List of K arrays — coefficient groups.

        Returns:
            Scalar or (S,) log-probability.
        """
        scales = self.compute_scales(R2, phi)
        return (
            self.log_prob_R2(R2)
            + self.log_prob_phi(phi)
            + self.log_prob_coefficients(coefficients, scales)
        )

    def sample_R2(self, key, sample_shape=()):
        """Sample R² from the Beta prior."""
        return self.R2_dist.sample(sample_shape=sample_shape, seed=key)

    def sample_phi(self, key, sample_shape=()):
        """Sample phi from the simplex prior."""
        return self.phi_dist.sample(sample_shape=sample_shape, seed=key)

    def initial_R2(self):
        """Initial value for R² (prior mean)."""
        return jnp.array(self.mu_R2, dtype=self.dtype)

    def initial_phi(self):
        """Initial value for phi (prior mean of Dirichlet)."""
        return jnp.array(
            self.component_weights / self.component_weights.sum(),
            dtype=self.dtype,
        )

    def bijectors(self):
        """Return bijectors for R² and phi for use in variational inference.

        Returns:
            dict with 'R2' and 'phi' bijector entries.
        """
        return {
            "R2": tfb.Sigmoid(),
            "phi": tfb.SoftmaxCentered(),
        }


# ── Numerical helpers ──

def _digamma(x):
    """Digamma function via JAX."""
    import jax
    return jax.lax.digamma(x)


def _trigamma(x):
    """Trigamma function via forward-mode AD on digamma."""
    import jax
    return jax.vmap(jax.grad(lambda xi: jax.lax.digamma(xi)))(x)
