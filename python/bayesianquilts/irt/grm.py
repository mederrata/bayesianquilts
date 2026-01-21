import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Callable
from bayesianquilts.metrics.ais import LikelihoodFunction, AutoDiffLikelihoodMixin

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
