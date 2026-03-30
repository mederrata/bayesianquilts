import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Optional, Callable

from bayesianquilts.metrics.ais import LikelihoodFunction, AutoDiffLikelihoodMixin
from bayesianquilts.distributions import AbsHorseshoe, SqrtInverseGamma
from bayesianquilts.vi.advi import build_factored_surrogate_posterior_generator
from bayesianquilts.irt.irt import IRTModel

from jax.scipy.special import xlogy
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import tf2jax as tf

class GradedResponseLikelihood(AutoDiffLikelihoodMixin, LikelihoodFunction):
    """
    Likelihood function for the Graded Response IRT Model.
    
    Likelihood is aggregated by person for person-level LOO-IC.
    """
    
    def __init__(self, dtype=jnp.float64):
        self.dtype = dtype

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """
        Compute log-likelihood for the Graded Response Model.

        Parameters (params):
            theta: person abilities (S, J) or (S, N_loo, J) for transformed params
            alpha: item discrimination (S, I) or (S, N_loo, I) for transformed params
            tau: item thresholds (S, I, K-1) or (S, N_loo, I, K-1) for transformed params

        Data (data):
            person_idx: indices for each observation (N_obs,)
            item_idx: indices for each observation (N_obs,)
            y: observed responses (0 to K-1) (N_obs,)
            n_people: total number of unique people (J)

        Returns:
            log_lik: logged likelihood aggregated by person
                     (S, J) for original params or (S, N_loo, J) for transformed params
        """
        person_idx = jnp.asarray(data["person_idx"], dtype=jnp.int32)
        item_idx = jnp.asarray(data["item_idx"], dtype=jnp.int32)
        y = jnp.asarray(data["y"], dtype=jnp.int32)
        n_people = data["n_people"]

        theta = jnp.asarray(params["theta"], dtype=self.dtype)
        # Allow alpha/tau to be in data (frozen) or params (active)
        alpha = jnp.asarray(params.get("alpha", data.get("alpha")), dtype=self.dtype)
        tau = jnp.asarray(params.get("tau", data.get("tau")), dtype=self.dtype)

        # Check if we have transformed params with extra N_loo dimension
        # Original: theta (S, J), alpha (S, I), tau (S, I, K-1)
        # Transformed: theta (S, N_loo, J), alpha (S, N_loo, I), tau (S, N_loo, I, K-1)
        has_loo_dim = theta.ndim == 3

        if has_loo_dim:
            # Transformed params: compute likelihood for each LOO case
            return self._log_likelihood_transformed(
                theta, alpha, tau, person_idx, item_idx, y, n_people
            )
        else:
            # Original params: standard computation
            return self._log_likelihood_standard(
                theta, alpha, tau, person_idx, item_idx, y, n_people
            )

    def _log_likelihood_standard(self, theta, alpha, tau, person_idx, item_idx, y, n_people):
        """Compute log-likelihood for standard (non-transformed) params."""
        S = theta.shape[0]
        N_obs = person_idx.shape[0]
        K_minus_1 = tau.shape[-1]

        # Gather parameters for each observation
        # theta_obs: (S, N_obs)
        theta_obs = jnp.take_along_axis(theta, person_idx[None, :], axis=1)
        # alpha_obs: (S, N_obs)
        alpha_obs = jnp.take_along_axis(alpha, item_idx[None, :], axis=1)
        # tau_obs: (S, N_obs, K-1)
        tau_obs = jnp.take_along_axis(tau, item_idx[None, :, None], axis=1)

        # Compute latent score: alpha * (theta - tau)
        # eta: (S, N_obs, K-1)
        eta = alpha_obs[:, :, None] * (theta_obs[:, :, None] - tau_obs)

        # Cumulative probabilities P(Y >= k) = sigmoid(eta)
        cum_probs = jax.nn.sigmoid(eta)

        # Pad cum_probs with 0 and 1 to get P(Y < k) for k in 0..K
        probs_le = jnp.concatenate([
            jnp.zeros((S, N_obs, 1), dtype=self.dtype),
            1.0 - cum_probs,
            jnp.ones((S, N_obs, 1), dtype=self.dtype)
        ], axis=-1)

        # P(Y = k) = P(Y < k+1) - P(Y < k)
        all_probs = jnp.diff(probs_le, axis=-1)

        # Select probability for observed y
        obs_probs = jnp.take_along_axis(all_probs, y[None, :, None], axis=2)
        obs_probs = jnp.squeeze(obs_probs, axis=2)

        log_lik_obs = jnp.log(jnp.clip(obs_probs, a_min=1e-15))

        # Aggregate by person
        def sum_by_person(ll_s):
            return jax.ops.segment_sum(ll_s, person_idx, num_segments=n_people)

        log_lik_person = jax.vmap(sum_by_person)(log_lik_obs)

        return log_lik_person

    def _log_likelihood_transformed(self, theta, alpha, tau, person_idx, item_idx, y, n_people):
        """
        Compute log-likelihood for transformed params with LOO dimension.

        For person-level LOO, when params have shape (S, N_loo, ...), we only need
        the likelihood of person n when using the transformation for LOO case n.
        This returns the diagonal: log_lik[s, n] = likelihood of person n under
        transformed params for leaving out person n.

        theta: (S, N_loo, J)
        alpha: (S, N_loo, I)
        tau: (S, N_loo, I, K-1)

        Returns: (S, N_loo) - log likelihood for each LOO case (diagonal extraction)
        """
        S, N_loo, J = theta.shape
        I = alpha.shape[2]
        K_minus_1 = tau.shape[-1]
        N_obs = person_idx.shape[0]

        def compute_for_loo_case(theta_s_n, alpha_s_n, tau_s_n, loo_idx):
            """Compute likelihood of person loo_idx under params for leaving them out."""
            # theta_s_n: (J,), alpha_s_n: (I,), tau_s_n: (I, K-1)
            # loo_idx: scalar - which person we're leaving out / computing for

            # Gather for observations
            theta_obs = theta_s_n[person_idx]  # (N_obs,)
            alpha_obs = alpha_s_n[item_idx]    # (N_obs,)
            tau_obs = tau_s_n[item_idx]        # (N_obs, K-1)

            # eta: (N_obs, K-1)
            eta = alpha_obs[:, None] * (theta_obs[:, None] - tau_obs)

            cum_probs = jax.nn.sigmoid(eta)

            probs_le = jnp.concatenate([
                jnp.zeros((N_obs, 1), dtype=self.dtype),
                1.0 - cum_probs,
                jnp.ones((N_obs, 1), dtype=self.dtype)
            ], axis=-1)

            all_probs = jnp.diff(probs_le, axis=-1)

            obs_probs = all_probs[jnp.arange(N_obs), y]
            log_lik_obs = jnp.log(jnp.clip(obs_probs, a_min=1e-15))

            # Aggregate by person
            log_lik_person = jax.ops.segment_sum(log_lik_obs, person_idx, num_segments=n_people)

            # Return only the likelihood for the LOO case person
            return log_lik_person[loo_idx]  # Scalar

        # For each LOO case n, compute likelihood of person n
        loo_indices = jnp.arange(N_loo)

        def compute_all_loo_for_sample(theta_s, alpha_s, tau_s):
            """Compute for all LOO cases within a sample."""
            # theta_s: (N_loo, J), alpha_s: (N_loo, I), tau_s: (N_loo, I, K-1)
            return jax.vmap(compute_for_loo_case)(theta_s, alpha_s, tau_s, loo_indices)

        # Vmap over samples
        result = jax.vmap(compute_all_loo_for_sample)(theta, alpha, tau)
        # result: (S, N_loo)

        return result

    def log_likelihood_gradient(self, data: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """
        Compute per-person gradient of log-likelihood w.r.t. parameters.

        For LOO-CV, we need the gradient of each person's log-likelihood separately.
        This returns gradients with shape (S, N, K) where N is the number of persons
        and K is the parameter dimension.

        Returns:
            Dict with:
                'theta': (S, J, J) - gradient of LL_n w.r.t. theta (mostly diagonal)
                'alpha': (S, J, I) - gradient of LL_n w.r.t. alpha
                'tau': (S, J, I, K-1) - gradient of LL_n w.r.t. tau
        """
        theta = params["theta"]  # (S, J)
        alpha = params.get("alpha", data.get("alpha"))
        tau = params.get("tau", data.get("tau"))

        n_people = data["n_people"]
        S = theta.shape[0]
        J = theta.shape[1]
        I = alpha.shape[1]
        K_minus_1 = tau.shape[2]

        # Compute gradient of each person's LL separately
        def single_sample_grads(th, al, ta):
            # th: (J,), al: (I,), ta: (I, K-1)

            def person_ll(t, a, tau_val, person_idx):
                # Compute LL for a single person
                p_in = {'theta': t[None, :], 'alpha': a[None, :], 'tau': tau_val[None, :]}
                ll = self.log_likelihood(data, p_in)  # (1, J)
                return ll[0, person_idx]  # Scalar: LL for this person

            def grad_for_person(person_idx):
                # Gradient of person_idx's LL w.r.t. all params
                grads = jax.grad(person_ll, argnums=(0, 1, 2))(th, al, ta, person_idx)
                return grads  # (theta_grad (J,), alpha_grad (I,), tau_grad (I, K-1))

            # Vectorize over all persons
            all_grads = jax.vmap(grad_for_person)(jnp.arange(n_people))
            # all_grads[0]: (J, J) - theta grads for each person
            # all_grads[1]: (J, I) - alpha grads for each person
            # all_grads[2]: (J, I, K-1) - tau grads for each person

            return {
                'theta': all_grads[0],  # (J, J)
                'alpha': all_grads[1],  # (J, I)
                'tau': all_grads[2]     # (J, I, K-1)
            }

        # Vmap over sample dimension S
        grads = jax.vmap(single_sample_grads)(theta, alpha, tau)
        # grads['theta']: (S, J, J)
        # grads['alpha']: (S, J, I)
        # grads['tau']: (S, J, I, K-1)

        return grads

    def log_likelihood_hessian_diag(self, data: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """
        Compute per-person diagonal of Hessian of log-likelihood w.r.t. parameters.

        For LOO-CV, we need the Hessian diagonal for each person's log-likelihood separately.

        Returns:
            Dict with:
                'theta': (S, J, J) - Hessian diagonal of LL_n w.r.t. theta
                'alpha': (S, J, I) - Hessian diagonal of LL_n w.r.t. alpha
                'tau': (S, J, I, K-1) - Hessian diagonal of LL_n w.r.t. tau
        """
        theta = params["theta"]  # (S, J)
        alpha = params.get("alpha", data.get("alpha"))
        tau = params.get("tau", data.get("tau"))

        n_people = data["n_people"]
        S = theta.shape[0]
        J = theta.shape[1]
        I = alpha.shape[1]
        K_minus_1 = tau.shape[2]

        def single_sample_hess_diag(th, al, ta):
            # th: (J,), al: (I,), ta: (I, K-1)

            def person_ll_theta(t, person_idx):
                p_in = {'theta': t[None, :], 'alpha': al[None, :], 'tau': ta[None, :]}
                ll = self.log_likelihood(data, p_in)
                return ll[0, person_idx]

            def person_ll_alpha(a, person_idx):
                p_in = {'theta': th[None, :], 'alpha': a[None, :], 'tau': ta[None, :]}
                ll = self.log_likelihood(data, p_in)
                return ll[0, person_idx]

            def person_ll_tau(t_val, person_idx):
                p_in = {'theta': th[None, :], 'alpha': al[None, :], 'tau': t_val[None, :]}
                ll = self.log_likelihood(data, p_in)
                return ll[0, person_idx]

            def hess_diag_for_person(person_idx):
                # Hessian diagonal for each parameter
                hess_theta = jax.hessian(person_ll_theta)(th, person_idx)
                hess_alpha = jax.hessian(person_ll_alpha)(al, person_idx)
                hess_tau = jax.hessian(person_ll_tau)(ta, person_idx)

                # Extract diagonals
                theta_diag = jnp.diag(hess_theta)  # (J,)
                alpha_diag = jnp.diag(hess_alpha)  # (I,)
                # For tau, flatten, get diag, reshape
                tau_flat_hess = hess_tau.reshape(I * K_minus_1, I * K_minus_1)
                tau_diag = jnp.diag(tau_flat_hess).reshape(I, K_minus_1)

                return theta_diag, alpha_diag, tau_diag

            # Vectorize over persons
            all_hess = jax.vmap(hess_diag_for_person)(jnp.arange(n_people))

            return {
                'theta': all_hess[0],  # (J, J)
                'alpha': all_hess[1],  # (J, I)
                'tau': all_hess[2]     # (J, I, K-1)
            }

        hess_diags = jax.vmap(single_sample_hess_diag)(theta, alpha, tau)
        return hess_diags



    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Flatten parameters into a single array for AIS.
        Expected keys: theta (S, J), alpha (S, I), tau (S, I, K-1)
        """
        theta = params["theta"]
        alpha = params["alpha"]
        tau = params["tau"]
        
        S = theta.shape[0]
        # Flatten all except S
        theta_flat = theta.reshape(S, -1)
        alpha_flat = alpha.reshape(S, -1)
        tau_flat = tau.reshape(S, -1)
        
        return jnp.concatenate([theta_flat, alpha_flat, tau_flat], axis=1)

    def reconstruct_parameters(self, flat_params: jnp.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct parameters from flattened array.
        """
        S = flat_params.shape[0]
        J = template["theta"].shape[1]
        I = template["alpha"].shape[1]
        K_minus_1 = template["tau"].shape[2]
        
        idx = 0
        theta = flat_params[:, idx:idx+J].reshape(S, J)
        idx += J
        alpha = flat_params[:, idx:idx+I].reshape(S, I)
        idx += I
        tau = flat_params[:, idx:idx+I*K_minus_1].reshape(S, I, K_minus_1)
        
        return {"theta": theta, "alpha": alpha, "tau": tau}


# =========================================================================
# GRModel - Graded Response Model (ported from autoencirt)
# =========================================================================

def make_shifted_softplus(min_value, hinge_softness=1.0, name="shifted_softplus"):
    """Creates a Softplus bijector with a specified minimum value."""
    return tfb.Chain(
        [tfb.Softplus(hinge_softness=hinge_softness), tfb.Shift(shift=min_value)],
        name=name,
    )


class GRModel(IRTModel):
    """Graded Response Model for IRT.

    Implements Samejima's Graded Response Model with hierarchical Bayesian priors.
    Supports optional stochastic imputation of missing responses via an imputation model.
    """

    response_type = "polytomous"

    def __init__(self, *args, **kwargs):
        super(GRModel, self).__init__(*args, **kwargs)
        self.create_distributions()

    def grm_model_prob(self, abilities, discriminations, difficulties):
        if self.include_independent:
            abilities = jnp.pad(
                abilities,
                [(0, 0)] * (len(discriminations.shape) - 3) + [(1, 0)] + [(0, 0)] * 2,
            )
        offsets = difficulties - abilities  # N x D x I x K-1
        scaled = offsets * discriminations
        logits = jax.nn.sigmoid(-scaled)
        logits = jnp.pad(
            logits,
            ([(0, 0)] * (len(logits.shape) - 1) + [(1, 0)]),
            mode="constant",
            constant_values=1,
        )
        logits = jnp.pad(
            logits,
            ([(0, 0)] * (len(logits.shape) - 1) + [(0, 1)]),
            mode="constant",
            constant_values=0,
        )
        probs = logits[..., :-1] - logits[..., 1:]

        # weight by discrimination
        weights = (
            jnp.abs(discriminations) ** self.weight_exponent
            / jnp.sum(jnp.abs(discriminations) ** self.weight_exponent, axis=-3)[
                ..., jnp.newaxis, :, :
            ]
        )
        probs = jnp.sum(probs * weights, axis=-3)
        return probs

    def grm_model_prob_d(
        self, abilities, discriminations, difficulties0, ddifficulties=None
    ):
        if ddifficulties is not None and ddifficulties.shape[-1] > 0:
            d0 = jnp.concat([difficulties0, ddifficulties], axis=-1)
        else:
            d0 = difficulties0
        difficulties = jnp.cumsum(d0, axis=-1)
        return self.grm_model_prob(abilities, discriminations, difficulties)

    def predictive_distribution(
        self,
        data,
        discriminations,
        difficulties0,
        abilities,
        ddifficulties=None,
        **kwargs
    ):
        if ddifficulties is not None and ddifficulties.shape[-1] > 0:
            ddifficulties = jnp.where(
                ddifficulties < 1e-1, 1e-1 * jnp.ones_like(ddifficulties), ddifficulties
            )
            difficulties = jnp.concat([difficulties0, ddifficulties], axis=-1)
        else:
            difficulties = difficulties0
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

        # grm_model_prob already applies discrimination weighting and
        # reduces over the dimensions axis (-3), so response_probs is 4D
        response_probs = self.grm_model_prob(abilities, discriminations, difficulties)

        rv_responses = tfd.Categorical(probs=response_probs)

        log_probs = rv_responses.log_prob(choices)

        imputation_pmfs = data.get('_imputation_pmfs')
        imputation_weights = data.get('_imputation_weights')
        if imputation_pmfs is not None:
            # Analytic Rao-Blackwellization: log[ sum_k q(k) * p(Y=k|phi) ]
            log_rp = jnp.log(jnp.maximum(response_probs, 1e-30))  # (S, N, I, K)
            log_q = jnp.log(jnp.maximum(imputation_pmfs, 1e-30))  # (N, I, K)
            rb = jax.scipy.special.logsumexp(
                log_rp + log_q[jnp.newaxis, ...], axis=-1
            )  # (S, N, I)

            if imputation_weights is not None:
                # Importance-sampling mode: weight RB contribution by
                # per-item stacking weight.  When w_mice=0, the missing
                # cell contributes 0 (ignorability).  When w_mice=1,
                # full Rao-Blackwellization with MICE PMFs.
                w = jnp.asarray(imputation_weights)  # (I,)
                weighted_rb = w[jnp.newaxis, jnp.newaxis, :] * rb  # (S, N, I)
                log_probs = jnp.where(
                    bad_choices[jnp.newaxis, ...], weighted_rb, log_probs
                )
            else:
                log_probs = jnp.where(
                    bad_choices[jnp.newaxis, ...], rb, log_probs
                )
        else:
            # Missing responses are marginalized out: sum_k p(k|theta) = 1,
            # so they contribute log(1) = 0 to the log-likelihood.
            log_probs = jnp.where(
                bad_choices[jnp.newaxis, ...], 0.0, log_probs
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
        abilities,
        ddifficulties=None,
        *args,
        **kwargs
    ):
        prediction = self.predictive_distribution(
            data,
            discriminations,
            difficulties0,
            abilities,
            ddifficulties=ddifficulties,
            *args,
            **kwargs
        )
        return prediction["log_likelihood"]

    def _prepare_ais_inputs(self, all_batches, samples):
        """Convert GRM batch data and surrogate samples to AIS format.

        Returns:
            (likelihood_fn, ais_data, ais_params) suitable for
            AdaptiveImportanceSampler.adaptive_is_loo.
        """
        # Build long-format data: (person_idx, item_idx, y)
        person_idxs = []
        item_idxs = []
        ys = []
        for batch in all_batches:
            people = np.asarray(batch[self.person_key], dtype=np.int32)
            for i, key in enumerate(self.item_keys):
                vals = np.asarray(batch[key])
                valid = (vals >= 0) & (vals < self.response_cardinality) & ~np.isnan(vals)
                person_idxs.append(people[valid])
                item_idxs.append(np.full(int(valid.sum()), i, dtype=np.int32))
                ys.append(vals[valid].astype(np.int32))

        ais_data = {
            'person_idx': np.concatenate(person_idxs),
            'item_idx': np.concatenate(item_idxs),
            'y': np.concatenate(ys),
            'n_people': self.num_people,
        }

        # Extract theta, alpha, tau from surrogate samples
        # abilities: (S, N, D, 1, 1) -> theta: (S, N)
        abilities = np.array(samples['abilities'])
        theta = abilities[:, :, 0, 0, 0]  # (S, N) assuming dim=1

        # discriminations: (S, 1, D, I, 1) -> alpha: (S, I)
        disc = np.array(samples['discriminations'])
        alpha = disc[:, 0, 0, :, 0]  # (S, I) assuming dim=1

        # difficulties: cumsum(concat(difficulties0, ddifficulties))
        diff0 = np.array(samples['difficulties0'])  # (S, 1, 1, I, 1)
        diff0 = diff0[:, 0, 0, :, 0]  # (S, I)
        if 'ddifficulties' in samples and samples['ddifficulties'].shape[-1] > 0:
            ddiff = np.array(samples['ddifficulties'])  # (S, 1, 1, I, K-2)
            ddiff = ddiff[:, 0, 0, :, :]  # (S, I, K-2)
            tau_raw = np.concatenate([diff0[:, :, np.newaxis], ddiff], axis=-1)
        else:
            tau_raw = diff0[:, :, np.newaxis]  # (S, I, 1)
        tau = np.cumsum(tau_raw, axis=-1)  # (S, I, K-1)

        ais_params = {
            'theta': jnp.array(theta),
            'alpha': jnp.array(alpha),
            'tau': jnp.array(tau),
        }

        likelihood_fn = GradedResponseLikelihood(dtype=jnp.float64)
        return likelihood_fn, ais_data, ais_params

    def create_distributions(self, grouping_params=None):
        """Create prior and surrogate distributions."""
        prior_scale = getattr(self, 'discrimination_prior_scale', None)
        prior_type = getattr(self, 'discrimination_prior', 'half_normal')

        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }

        self.bijectors["discriminations"] = tfb.Softplus()
        if self.response_cardinality > 2:
            self.bijectors["ddifficulties"] = tfb.Softplus()

        if prior_type == 'horseshoe':
            self.bijectors["global_scale"] = tfb.Softplus()

        K = self.response_cardinality

        # Center mu prior at -(K-2)/2 so the median threshold starts
        # near 0 after cumsum with ddifficulties ~ HalfNormal(1).
        d0_loc = -(K - 2) / 2.0
        grm_joint_distribution_dict = dict(
            mu=tfd.Independent(
                tfd.Normal(
                    loc=jnp.full(
                        (1, self.dimensions, self.num_items, 1), d0_loc, dtype=self.dtype
                    ),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
            difficulties0=lambda mu: tfd.Independent(
                tfd.Normal(
                    loc=jnp.asarray(mu, dtype=self.dtype),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
        )

        if prior_type == 'horseshoe':
            # TFP Horseshoe with local shrinkage marginalized analytically.
            # Global shrinkage τ ~ HalfCauchy is kept as a separate parameter.
            # discriminations | τ ~ AbsHorseshoe(scale=τ) for positive,
            #                      Horseshoe(scale=τ) otherwise.
            global_scale_prior = float(prior_scale) if prior_scale is not None else 1.0
            grm_joint_distribution_dict["global_scale"] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros((1, self.dimensions, 1, 1), dtype=self.dtype),
                    scale=jnp.asarray(global_scale_prior, dtype=self.dtype)
                    * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            )
            if self.positive_discriminations:
                grm_joint_distribution_dict["discriminations"] = (
                    lambda global_scale: tfd.Independent(
                        AbsHorseshoe(
                            scale=jnp.asarray(global_scale, dtype=self.dtype)
                            * jnp.ones(
                                (1, self.dimensions, self.num_items, 1),
                                dtype=self.dtype,
                            ),
                        ),
                        reinterpreted_batch_ndims=4,
                    )
                )
            else:
                grm_joint_distribution_dict["discriminations"] = (
                    lambda global_scale: tfd.Independent(
                        tfd.Horseshoe(
                            scale=jnp.asarray(global_scale, dtype=self.dtype)
                            * jnp.ones(
                                (1, self.dimensions, self.num_items, 1),
                                dtype=self.dtype,
                            ),
                        ),
                        reinterpreted_batch_ndims=4,
                    )
                )
        else:
            # Simple prior on discriminations — no hyperpriors
            _scale = float(prior_scale) if prior_scale is not None else 2.0
            disc_scale = jnp.asarray(
                _scale * jnp.ones(
                    (1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                dtype=self.dtype,
            )
            if prior_type == 'half_cauchy':
                grm_joint_distribution_dict["discriminations"] = tfd.Independent(
                    tfd.HalfCauchy(
                        loc=jnp.zeros_like(disc_scale),
                        scale=disc_scale,
                    ),
                    reinterpreted_batch_ndims=4,
                )
            else:
                grm_joint_distribution_dict["discriminations"] = tfd.Independent(
                    tfd.HalfNormal(scale=disc_scale),
                    reinterpreted_batch_ndims=4,
                )
        if self.response_cardinality > 2:
            grm_joint_distribution_dict["ddifficulties"] = tfd.Independent(
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
            )
        if grouping_params is not None:
            grm_joint_distribution_dict["probs"] = tfd.Independent(
                tfd.Dirichlet(jnp.astype(grouping_params, self.dtype)),
                reinterpreted_batch_ndims=1,
            )
            grm_joint_distribution_dict["mu_ability"] = lambda sigma: tfd.Independent(
                tfd.Normal(
                    loc=jnp.zeros((self.dimensions, self.num_groups), self.dtype),
                    scale=jnp.asarray(sigma, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=2,
            )
            self.bijectors["sigma"] = tfb.Softplus()
            grm_joint_distribution_dict["sigma"] = tfd.Independent(
                tfd.HalfNormal(
                    scale=0.5 * jnp.ones((self.dimensions, self.num_groups), self.dtype)
                ),
                reinterpreted_batch_ndims=2,
            )

            grm_joint_distribution_dict["abilities"] = (
                lambda probs, mu_ability, sigma: tfd.Independent(
                    tfd.Mixture(
                        cat=tfd.Categorical(probs=probs),
                        components=[
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        jnp.squeeze(
                                            mu_ability[..., jnp.newaxis, :, 0:1]
                                            + jnp.zeros(
                                                shape=(
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                dtype=self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                    scale=(
                                        jnp.squeeze(
                                            sigma[..., jnp.newaxis, :, 0:1]
                                            + jnp.zeros(
                                                shape=(
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                dtype=self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                ),
                                reinterpreted_batch_ndims=3,
                            ),
                            tfd.Independent(
                                tfd.Normal(
                                    loc=(
                                        jnp.squeeze(
                                            mu_ability[..., jnp.newaxis, :, 1:2]
                                            + jnp.zeros(
                                                (
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                    scale=(
                                        jnp.squeeze(
                                            sigma[..., jnp.newaxis, :, 1:2]
                                            + jnp.zeros(
                                                (
                                                    1,
                                                    self.num_people,
                                                    (
                                                        self.dimensions
                                                        if not self.include_independent
                                                        else self.dimensions - 1
                                                    ),
                                                    1,
                                                ),
                                                self.dtype,
                                            )
                                        )
                                    )[..., jnp.newaxis, jnp.newaxis],
                                ),
                                reinterpreted_batch_ndims=3,
                            ),
                        ],
                    ),
                    reinterpreted_batch_ndims=1,
                )
            )
        else:
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
                parameterization=self.parameterization,
                rank=self.rank,
            )
        )
        self.params = self.surrogate_parameter_initializer()

    def score(self, responses, samples=400, mm_iterations=10):
        responses = jnp.astype(responses, jnp.int32)
        sampling_rv = tfd.Independent(
            tfd.Normal(
                loc=jnp.mean(self.calibrated_expectations["abilities"], axis=0),
                scale=jnp.std(
                    self.calibrated_expectations["abilities"], axis=0
                ),
            ),
            reinterpreted_batch_ndims=2,
        )
        trait_samples = sampling_rv.sample(samples)
        sample_log_p = sampling_rv.log_prob(trait_samples)

        ddiff = (
            jnp.expand_dims(self.surrogate_sample["ddifficulties"], 0)
            if "ddifficulties" in self.surrogate_sample
            else None
        )
        response_probs = self.grm_model_prob_d(
            abilities=trait_samples[..., jnp.newaxis, jnp.newaxis, :, :, :],
            discriminations=jnp.expand_dims(self.surrogate_sample["discriminations"], 0),
            difficulties0=jnp.expand_dims(self.surrogate_sample["difficulties0"], 0),
            ddifficulties=ddiff,
        )

        response_probs = jnp.mean(response_probs, axis=-4)

        response_rv = tfd.Independent(
            tfd.Categorical(probs=response_probs), reinterpreted_batch_ndims=1
        )
        lp = response_rv.log_prob(responses)
        l_w = lp[..., jnp.newaxis] - sample_log_p[:, jnp.newaxis, :]
        w = jnp.exp(l_w) / jnp.sum(jnp.exp(l_w), axis=0, keepdims=True)
        mean = jnp.sum(w * trait_samples[:, jnp.newaxis, :, 0, 0], axis=0)
        mean2 = jnp.sum(
            w * trait_samples[:, jnp.newaxis, :, 0, 0] ** 2, axis=0
        )
        std = jnp.sqrt(mean2 - mean**2)
        return mean, std, w, trait_samples

    def fit_dim(self, *args, dim: int, **kwargs):
        if dim >= self.dimensions:
            raise ValueError("Dimension to fit must be less than model dimensions")
        optimizing_keys = [
            k
            for k in self.params.keys()
            if (
                not any(
                    k.startswith(prefix)
                    and not k.startswith(f"{prefix}{dim}")
                    for prefix in [
                        "discriminations_",
                        "ddifficulties_",
                        "difficulties0_",
                        "kappa_",
                        "kappa_a_",
                    ]
                )
            )
        ]
        return self.fit(*args, **kwargs, optimize_keys=optimizing_keys)

    def unormalized_log_prob(self, data, prior_weight=1., **params):
        log_prior = self.joint_prior_distribution.log_prob(params)
        prediction = self.predictive_distribution(data, **params)
        log_likelihood = prediction["log_likelihood"]
        weights = prediction["discriminations"]
        weights = weights / jnp.sum(weights, axis=-3, keepdims=True)
        if "eta" in params:
            entropy = -xlogy(weights, weights) / params["eta"]
            entropy = jnp.sum(entropy, axis=[-1, -2, -3, -4])
        else:
            entropy = jnp.zeros_like(log_prior)

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
        if 'sample_weights' in data:
            sw = jnp.asarray(data['sample_weights'], dtype=log_likelihood.dtype)
            weighted_ll = jnp.sum(sw[None, :] * log_likelihood, axis=-1)
        else:
            weighted_ll = jnp.sum(log_likelihood, axis=-1)
        return jnp.astype(prior_weight, log_prior.dtype) * (log_prior - entropy) + weighted_ll

    def marginal_log_prob(self, data, theta_grid=None, prior_weight=1.0,
                          **item_params):
        """Rao-Blackwellized log posterior over item parameters only.

        Integrates out person abilities on a grid, yielding the marginal
        log posterior over discriminations and difficulties:

            log p(Xi | data) = log p(Xi) + sum_p log int P(x_p | theta, Xi) pi(theta) dtheta

        Args:
            data: Data dict with item response columns and person key.
            theta_grid: 1D array of quadrature points for ability integration.
                Defaults to 200 points on [-6, 6].
            prior_weight: Weight on the log prior.
            **item_params: Item parameters (discriminations, difficulties0,
                ddifficulties, mu). Must NOT include abilities.

        Returns:
            Scalar log marginal posterior.
        """
        if theta_grid is None:
            theta_grid = jnp.linspace(-6, 6, 200, dtype=self.dtype)

        n_grid = len(theta_grid)
        people = data[self.person_key].astype(jnp.int32)
        n_people_batch = len(people)

        # Build abilities tensor: (n_grid, n_people, 1, 1, 1)
        # Each grid point is a "sample" — abilities are constant across people
        # within each grid point (shared ability for integration)
        # Shape: (n_grid, n_people, dim=1, 1, 1)
        abilities_grid = jnp.broadcast_to(
            theta_grid[:, None, None, None, None],
            (n_grid, n_people_batch, 1, 1, 1)
        )

        # Prepare difficulties
        discriminations = item_params['discriminations']
        difficulties0 = item_params['difficulties0']
        ddifficulties = item_params.get('ddifficulties')

        if ddifficulties is not None and ddifficulties.shape[-1] > 0:
            ddifficulties_safe = jnp.where(
                ddifficulties < 1e-1,
                1e-1 * jnp.ones_like(ddifficulties),
                ddifficulties
            )
            difficulties = jnp.concat([difficulties0, ddifficulties_safe], axis=-1)
        else:
            difficulties = difficulties0
        difficulties = jnp.cumsum(difficulties, axis=-1)

        # Compute P(x_pi = k | theta, Xi) for all grid points
        # grm_model_prob expects: abilities (n_grid, n_people, D, 1, 1)
        #                         disc     (1, D, I, 1)
        #                         diff     (1, D, I, K-1)
        response_probs = self.grm_model_prob(
            abilities_grid, discriminations, difficulties
        )  # (n_grid, n_people, I, K)

        # Get observed responses
        choices = jnp.concat(
            [data[k][:, jnp.newaxis] for k in self.item_keys], axis=-1
        )  # (n_people, I)
        bad = (choices < 0) | (choices >= self.response_cardinality) | jnp.isnan(choices)
        choices_safe = jnp.where(bad, jnp.zeros_like(choices), choices)

        # P(observed response | theta, Xi) per person per item
        # response_probs: (n_grid, n_people, I, K)
        # choices_safe: (n_people, I) -> (1, n_people, I, 1)
        obs_probs = jnp.take_along_axis(
            response_probs,
            choices_safe[None, :, :, None].astype(jnp.int32),
            axis=-1
        ).squeeze(-1)  # (n_grid, n_people, I)

        # For missing items, set prob to 1 (ignorable)
        obs_probs = jnp.where(bad[None, :, :], 1.0, obs_probs)

        # Log-likelihood per person per grid point
        # sum over items, then logsumexp over grid
        log_lik_per_grid = jnp.sum(
            jnp.log(jnp.clip(obs_probs, 1e-30, None)), axis=-1
        )  # (n_grid, n_people)

        # Prior on theta: N(0, 1)
        log_prior_theta = -0.5 * theta_grid ** 2 - 0.5 * jnp.log(2 * jnp.pi)
        # Add log prior and log grid spacing
        dtheta = theta_grid[1] - theta_grid[0]
        log_integrand = log_lik_per_grid + log_prior_theta[:, None] + jnp.log(dtheta)

        # Marginal log-likelihood per person: log sum_theta exp(log_integrand)
        marginal_ll_per_person = jax.scipy.special.logsumexp(
            log_integrand, axis=0
        )  # (n_people,)

        # Handle sample weights
        if 'sample_weights' in data:
            sw = jnp.asarray(data['sample_weights'], dtype=marginal_ll_per_person.dtype)
            total_marginal_ll = jnp.sum(sw * marginal_ll_per_person)
        else:
            total_marginal_ll = jnp.sum(marginal_ll_per_person)

        # Log prior on item parameters
        # Build a dict with just item params for the prior
        full_params = dict(item_params)
        # Add dummy abilities for the joint prior (it expects all params)
        full_params['abilities'] = jnp.zeros(
            (self.num_people, 1, 1, 1), dtype=self.dtype
        )
        log_prior_items = self.joint_prior_distribution.log_prob(full_params)

        return prior_weight * log_prior_items + total_marginal_ll

    def fit_marginal_mcmc(
        self,
        data,
        theta_grid=None,
        num_chains=4,
        num_warmup=500,
        num_samples=500,
        target_accept_prob=0.85,
        max_tree_depth=10,
        step_size=0.01,
        seed=None,
        verbose=True,
    ):
        """Run NUTS on item parameters with abilities Rao-Blackwellized out.

        Targets the marginal posterior over item parameters (discriminations,
        difficulties) by integrating out person abilities on a grid. Much
        lower-dimensional than joint MCMC.

        Args:
            data: Full data dict (all people).
            theta_grid: Quadrature grid for ability integration.
            num_chains: Number of NUTS chains.
            num_warmup: Warmup steps per chain.
            num_samples: Post-warmup samples per chain.
            target_accept_prob: NUTS target acceptance.
            max_tree_depth: Max NUTS tree depth.
            step_size: Initial step size.
            seed: Random seed.
            verbose: Print progress.

        Returns:
            Dict mapping item param names to arrays of shape
            (num_chains, num_samples, ...).
        """
        import jax
        from jax import random
        import tensorflow_probability.substrates.jax.mcmc as tfmcmc

        if theta_grid is None:
            theta_grid = jnp.linspace(-6, 6, 200, dtype=self.dtype)

        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        # Item parameter names (everything except abilities)
        item_var_list = [v for v in self.var_list if v != 'abilities']

        if verbose:
            print(f"Marginal MCMC (Rao-Blackwellized abilities)")
            print(f"  Item params: {item_var_list}")
            n_params = sum(
                np.prod(self.joint_prior_distribution.sample(
                    seed=key)[v].shape)
                for v in item_var_list
            )
            print(f"  Total dimensions: {n_params}")
            print(f"  Theta grid: {len(theta_grid)} points")
            print(f"  Chains: {num_chains}, Warmup: {num_warmup}, "
                  f"Samples: {num_samples}")

        # Target log prob: marginal over abilities
        def target_log_prob_fn(*params_flat):
            params_dict = {k: v for k, v in zip(item_var_list, params_flat)}
            return self.marginal_log_prob(
                data, theta_grid=theta_grid, **params_dict
            )

        # Initialize from ADVI solution if available
        key, init_key = random.split(key)
        if self.params is not None:
            if verbose:
                print("  Initializing from ADVI solution")
            initial_states = {}
            for var in item_var_list:
                # Find the loc parameter in the surrogate
                loc_key = None
                for pk in self.params:
                    if pk.startswith(var) and pk.endswith('loc'):
                        loc_key = pk
                        break
                if loc_key is not None:
                    val = self.params[loc_key]
                    # Tile across chains with small perturbation
                    initial_states[var] = jnp.stack([
                        val + 0.01 * random.normal(
                            random.fold_in(init_key, c), val.shape)
                        for c in range(num_chains)
                    ], axis=0)
                else:
                    prior_sample = self.joint_prior_distribution.sample(
                        seed=random.fold_in(init_key, 0))
                    initial_states[var] = jnp.stack([
                        prior_sample[var] + 0.01 * random.normal(
                            random.fold_in(init_key, c),
                            prior_sample[var].shape)
                        for c in range(num_chains)
                    ], axis=0)
        else:
            if verbose:
                print("  Initializing from prior")
            initial_states = {}
            for var in item_var_list:
                samples = self.joint_prior_distribution.sample(
                    num_chains, seed=init_key)
                initial_states[var] = samples[var]

        # Run NUTS per chain
        all_samples = {var: [] for var in item_var_list}
        all_accept = []

        for chain_idx in range(num_chains):
            key, chain_key = random.split(key)
            if verbose:
                print(f"\n  Chain {chain_idx + 1}/{num_chains}:")

            chain_init = [initial_states[var][chain_idx]
                          for var in item_var_list]

            nuts_kernel = tfmcmc.NoUTurnSampler(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                max_tree_depth=max_tree_depth,
            )
            adaptive_kernel = tfmcmc.DualAveragingStepSizeAdaptation(
                inner_kernel=nuts_kernel,
                num_adaptation_steps=num_warmup,
                target_accept_prob=target_accept_prob,
            )

            @jax.jit
            def run_chain(init_state, seed):
                samples, kernel_results = tfmcmc.sample_chain(
                    num_results=num_samples,
                    num_burnin_steps=num_warmup,
                    current_state=init_state,
                    kernel=adaptive_kernel,
                    seed=seed,
                    trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
                )
                return samples, kernel_results

            if verbose:
                print("    Running warmup and sampling...")

            samples, is_accepted = run_chain(chain_init, chain_key)

            accept_ratio = float(jnp.mean(is_accepted))
            if verbose:
                print(f"    Acceptance ratio: {accept_ratio:.3f}")

            if accept_ratio > 1e-6:
                for var, sample in zip(item_var_list, samples):
                    all_samples[var].append(sample)
                all_accept.append(accept_ratio)
            else:
                if verbose:
                    print(f"    Chain discarded (acceptance too low)")

        # Stack chains
        result = {}
        for var in item_var_list:
            if all_samples[var]:
                result[var] = jnp.stack(all_samples[var], axis=0)

        if verbose:
            print(f"\n  Mean acceptance: {np.mean(all_accept):.3f}")
            for var in item_var_list:
                if var in result:
                    flat = result[var].reshape(-1, *result[var].shape[2:])
                    print(f"  {var}: mean={jnp.mean(flat):.4f}, "
                          f"std={jnp.std(flat):.4f}")

        self.mcmc_samples = result
        return result
