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
        prior_type = getattr(self, 'discrimination_prior', 'horseshoe')
        expected_sparsity = getattr(self, 'expected_sparsity', None)
        use_reg_horseshoe = (prior_type == 'regularized_horseshoe'
                             or (prior_type == 'horseshoe' and expected_sparsity is not None))
        use_horseshoe = (not use_reg_horseshoe
                         and prior_scale is None and prior_type == 'horseshoe')

        self.bijectors = {
            k: tfb.Identity() for k in ["abilities", "mu", "difficulties0"]
        }

        self.bijectors["discriminations"] = tfb.Softplus()
        if self.response_cardinality > 2:
            self.bijectors["ddifficulties"] = tfb.Softplus()

        if use_horseshoe:
            self.bijectors["eta"] = tfb.Softplus()
            self.bijectors["kappa"] = tfb.Softplus()
            self.bijectors["eta_a"] = tfb.Softplus()
            self.bijectors["kappa_a"] = tfb.Softplus()
            self.bijectors["xi"] = tfb.Softplus()

        if use_reg_horseshoe:
            self.bijectors["local_scales"] = tfb.Softplus()
            self.bijectors["global_scale"] = tfb.Softplus()
            self.bijectors["slab_scale2"] = tfb.Softplus()

        K = self.response_cardinality

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
                    loc=jnp.asarray(mu, dtype=self.dtype),
                    scale=jnp.ones(
                        (1, self.dimensions, self.num_items, 1), dtype=self.dtype
                    ),
                ),
                reinterpreted_batch_ndims=4,
            ),
        )

        if use_reg_horseshoe:
            # Regularized horseshoe (Piironen & Vehtari, 2017)
            # τ₀ = (p₀ / (p - p₀)) / √n
            p = self.num_items
            if expected_sparsity is not None:
                p0 = expected_sparsity * p
            else:
                p0 = p / 10.0  # default: 1/10 of items are relevant
            tau0 = (p0 / (p - p0)) / jnp.sqrt(float(self.num_people))

            slab_scale = getattr(self, 'slab_scale', 2.0)
            slab_df = getattr(self, 'slab_df', 4)

            # Local scales: λ_i ~ C+(0, 1) per item
            grm_joint_distribution_dict["local_scales"] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros((1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                    scale=jnp.ones((1, self.dimensions, self.num_items, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            )
            # Global scale: τ ~ C+(0, τ₀)
            grm_joint_distribution_dict["global_scale"] = tfd.Independent(
                tfd.HalfCauchy(
                    loc=jnp.zeros((1, self.dimensions, 1, 1), dtype=self.dtype),
                    scale=jnp.asarray(tau0, dtype=self.dtype)
                    * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            )
            # Slab: c² ~ InverseGamma(ν/2, ν·s²/2)
            grm_joint_distribution_dict["slab_scale2"] = tfd.Independent(
                tfd.InverseGamma(
                    jnp.asarray(slab_df / 2.0, dtype=self.dtype)
                    * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    jnp.asarray(slab_df * slab_scale**2 / 2.0, dtype=self.dtype)
                    * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            )
            # Discriminations: α ~ HalfNormal(τ·λ̃)
            # where λ̃ = c·λ / √(c² + τ²·λ²)
            def _reg_horseshoe_disc(local_scales, global_scale, slab_scale2):
                ls = jnp.asarray(local_scales, dtype=self.dtype)
                gs = jnp.asarray(global_scale, dtype=self.dtype)
                c2 = jnp.asarray(slab_scale2, dtype=self.dtype)
                c = jnp.sqrt(c2)
                tau2_lambda2 = gs**2 * ls**2
                lambda_tilde = c * ls / jnp.sqrt(c2 + tau2_lambda2)
                eff_scale = jnp.maximum(gs * lambda_tilde, jnp.asarray(1e-10, dtype=self.dtype))
                return tfd.Independent(
                    tfd.HalfNormal(scale=eff_scale),
                    reinterpreted_batch_ndims=4,
                )
            grm_joint_distribution_dict["discriminations"] = _reg_horseshoe_disc

        elif use_horseshoe:
            grm_joint_distribution_dict["discriminations"] = (
                (
                    lambda eta, kappa: tfd.Independent(
                        AbsHorseshoe(scale=jnp.asarray(eta, dtype=self.dtype) * jnp.asarray(kappa, dtype=self.dtype)), reinterpreted_batch_ndims=4
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
                            scale=jnp.asarray(eta, dtype=self.dtype) * jnp.asarray(kappa, dtype=self.dtype),
                        ),
                        reinterpreted_batch_ndims=4,
                    )
                )
            )
            grm_joint_distribution_dict["eta"] = tfd.Independent(
                tfd.HalfNormal(
                    scale=self.eta_scale
                    * jnp.ones((1, 1, self.num_items, 1), dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            )
            grm_joint_distribution_dict["kappa"] = lambda kappa_a: tfd.Independent(
                SqrtInverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    jnp.asarray(1.0, dtype=self.dtype) / jnp.asarray(kappa_a, dtype=self.dtype),
                ),
                reinterpreted_batch_ndims=4,
            )
            grm_joint_distribution_dict["kappa_a"] = tfd.Independent(
                tfd.InverseGamma(
                    0.5 * jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype),
                    jnp.ones((1, self.dimensions, 1, 1), dtype=self.dtype)
                    / self.kappa_scale**2,
                ),
                reinterpreted_batch_ndims=4,
            )
        else:
            # Simple prior on discriminations — no hyperpriors
            _scale = float(prior_scale) if prior_scale is not None else 1.0
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
        return jnp.astype(prior_weight, log_prior.dtype) * (log_prior - entropy) + jnp.sum(
            log_likelihood, axis=-1
        )
