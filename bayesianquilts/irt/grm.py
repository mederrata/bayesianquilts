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
            theta: person abilities (S, J)
            alpha: item discrimination (S, I)
            tau: item thresholds (S, I, K-1)
            
        Data (data):
            person_idx: indices for each observation (N,)
            item_idx: indices for each observation (N,)
            y: observed responses (0 to K-1) (N,)
            n_people: total number of unique people (J)
            
        Returns:
            log_lik: logged likelihood aggregated by person (S, J)
        """
        person_idx = jnp.asarray(data["person_idx"], dtype=jnp.int32)
        item_idx = jnp.asarray(data["item_idx"], dtype=jnp.int32)
        y = jnp.asarray(data["y"], dtype=jnp.int32)
        n_people = data["n_people"]
        
        theta = params["theta"]  # (S, J)
        # Allow alpha/tau to be in data (frozen) or params (active)
        alpha = params.get("alpha", data.get("alpha"))  # (S, I)
        tau = params.get("tau", data.get("tau"))      # (S, I, K-1)
        
        S = theta.shape[0]
        N = person_idx.shape[0]
        K_minus_1 = tau.shape[-1]
        
        # Gather parameters for each observation
        # theta_obs: (S, N)
        theta_obs = jnp.take_along_axis(theta, person_idx[None, :], axis=1)
        # alpha_obs: (S, N)
        alpha_obs = jnp.take_along_axis(alpha, item_idx[None, :], axis=1)
        # tau_obs: (S, N, K-1)
        tau_obs = jnp.take_along_axis(tau, item_idx[None, :, None], axis=1)
        
        # Compute latent score: alpha * (theta - tau)
        # eta: (S, N, K-1)
        eta = alpha_obs[:, :, None] * (theta_obs[:, :, None] - tau_obs)
        
        # Cumulative probabilities P(Y >= k) = sigmoid(eta)
        # cum_probs: (S, N, K-1)
        cum_probs = jax.nn.sigmoid(eta)
        
        # Category probabilities P(Y = k)
        # For k=0: 1 - P(Y >= 1)
        # For 0 < k < K-1: P(Y >= k) - P(Y >= k+1)
        # For k=K-1: P(Y >= K-1)
        
        # Pad cum_probs with 1 and 0 to simplify calculation
        # Shape: (S, N, K+1) where K = (K-1) + 1
        probs_le = jnp.concatenate([
            jnp.zeros((S, N, 1), dtype=self.dtype),
            1.0 - cum_probs, # P(Y < k)
            jnp.ones((S, N, 1), dtype=self.dtype)
        ], axis=-1)
        
        # This is P(Y < k) where k ranges from 0 to K
        # P(Y = k) = P(Y < k+1) - P(Y < k)
        # P(Y = 0) = P(Y < 1) - P(Y < 0) = (1 - P(Y >= 1)) - 0 = 1 - P(Y >= 1). OK.
        # P(Y = K-1) = P(Y < K) - P(Y < K-1) = 1 - (1 - P(Y >= K-1)) = P(Y >= K-1). OK.
        
        all_probs = jnp.diff(probs_le, axis=-1) # (S, N, K)
        
        # Select the probability for the observed y
        # y is (N,) with values in 0..K-1
        obs_probs = jnp.take_along_axis(all_probs, y[None, :, None], axis=2)
        obs_probs = jnp.squeeze(obs_probs, axis=2) # (S, N)
        
        log_lik_obs = jnp.log(jnp.clip(obs_probs, a_min=1e-15)) # (S, N)
        
        # Aggregate by person
        # Use segment_sum to sum over observations for each person
        # We need to broadcast across S
        def sum_by_person(ll_s):
            return jax.ops.segment_sum(ll_s, person_idx, num_segments=n_people)
            
        log_lik_person = jax.vmap(sum_by_person)(log_lik_obs) # (S, J)
        
        return log_lik_person

    def log_likelihood_gradient(self, data: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """
        Compute gradient of log-likelihood w.r.t. parameters.
        Overridden to handle 'alpha' and 'tau' potentially being in 'data' (frozen)
        and to ensure correct slicing over the sample dimension S.
        """
        theta = params["theta"]  # (S, J)
        # alpha/tau might be in data or params. If in data, they are (S, I, ...)
        alpha = params.get("alpha", data.get("alpha"))
        tau = params.get("tau", data.get("tau"))
        
        # We define a function for a SINGLE sample s that maps theta_s -> LL_s
        def single_sample_grad(th, al, ta):
            # th: (J,)
            # al: (I,)
            # ta: (I, K-1)
            
            def scalar_ll_func(t, a, tau_val):
                # Construct inputs for log_likelihood
                p_in = {'theta': t[None, :]}
                # Pass alpha/tau in params so they are used
                p_in['alpha'] = a[None, :]
                p_in['tau'] = tau_val[None, :]
                
                # Data should not contain alpha/tau if we want to be sure, 
                # but params takes precedence anyway.
                # Use data from closure
                
                # log_likelihood returns (S, N) -> (1, J)
                ll = self.log_likelihood(data, p_in)
                return jnp.sum(ll) # Sum over people to get scalar target
            
            # Compute gradients of scalar target sum(LL)
            grads = jax.grad(scalar_ll_func, argnums=(0, 1, 2))(th, al, ta)
            return {'theta': grads[0], 'alpha': grads[1], 'tau': grads[2]}
            
        # Vmap over the sample dimension S
        grads = jax.vmap(single_sample_grad)(theta, alpha, tau)
        
        return grads

    def log_likelihood_hessian_diag(self, data: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """
        Compute diagonal of Hessian of log-likelihood w.r.t. parameters.
        Overridden to handle 'alpha' and 'tau' being in 'data' (frozen).
        """
        theta = params["theta"]  # (S, J)
        alpha = params.get("alpha", data.get("alpha"))
        tau = params.get("tau", data.get("tau"))
        
        def single_sample_hess_diag(th, al, ta):
            def f(t):
                p_in = {'theta': t[None, :]}
                d_in = data.copy()
                d_in['alpha'] = al[None, :]
                d_in['tau'] = ta[None, :]
                ll = self.log_likelihood(d_in, p_in)
                return ll[0] # (J,)

            # We want diagonal of Hessian.
            # L = sum(LL_j). But LL_j only depends on theta_j.
            # So Hessian is diagonal.
            # We can compute full Hessian and take diagonal (since J is small, = batch_size).
            
            # Note: f(t) returns (J,) vector of likelihoods.
            # We want Hessian of SCALAR sum(LL) w.r.t. theta?
            # Or Hessian of LL vector?
            # ais.py log_likelihood_hessian_diag expects "Hessian diagonal matching params ... + (N,) dim"?
            # No, looking at ais.py implementations:
            # "hess_diag_mu = -sigma * (1-sigma)" (S, N)
            # It returns the diagonal elements of the Hessian matrix of the log likelihood w.r.t parameters.
            # Usually for independent data, Cross terms are zero.
            # For theta_j, d^2 L / d theta_j^2 is the j-th diagonal element.
            
            # So we sum f(t) to get scalar L, then hessian.
            def scalar_f(t):
                return jnp.sum(f(t))
                
            hess = jax.hessian(scalar_f)(th) # (J, J)
            return {'theta': jnp.diag(hess)} # (J,)

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
