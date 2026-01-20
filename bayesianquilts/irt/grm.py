import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Optional, Callable
from bayesianquilts.metrics.ais import LikelihoodFunction, AutoDiffLikelihoodMixin

class GradedResponseLikelihood(AutoDiffLikelihoodMixin, LikelihoodFunction):
    """
    Likelihood function for the Graded Response IRT Model.
    
    Likelihood is aggregated by person (climber) for person-level LOO-IC.
    """
    
    def __init__(self, dtype=jnp.float64):
        self.dtype = dtype

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """
        Compute log-likelihood for the Graded Response Model.
        
        Parameters (params):
            theta: climber abilities (S, J)
            alpha: boulder discrimination (S, I)
            tau: boulder thresholds (S, I, K-1)
            
        Data (data):
            climber_idx: indices for each observation (N,)
            boulder_idx: indices for each observation (N,)
            y: observed responses (0 to K-1) (N,)
            n_climbers: total number of unique climbers (J)
            
        Returns:
            log_lik: logged likelihood aggregated by climber (S, J)
        """
        climber_idx = jnp.asarray(data["climber_idx"], dtype=jnp.int32)
        boulder_idx = jnp.asarray(data["boulder_idx"], dtype=jnp.int32)
        y = jnp.asarray(data["y"], dtype=jnp.int32)
        n_climbers = data["n_climbers"]
        
        theta = params["theta"]  # (S, J)
        alpha = params["alpha"]  # (S, I)
        tau = params["tau"]      # (S, I, K-1)
        
        S = theta.shape[0]
        N = climber_idx.shape[0]
        K_minus_1 = tau.shape[-1]
        
        # Gather parameters for each observation
        # theta_obs: (S, N)
        theta_obs = jnp.take_along_axis(theta, climber_idx[None, :], axis=1)
        # alpha_obs: (S, N)
        alpha_obs = jnp.take_along_axis(alpha, boulder_idx[None, :], axis=1)
        # tau_obs: (S, N, K-1)
        tau_obs = jnp.take_along_axis(tau, boulder_idx[None, :, None], axis=1)
        
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
        
        # Aggregate by climber
        # Use segment_sum to sum over observations for each climber
        # We need to broadcast across S
        def sum_by_climber(ll_s):
            return jax.ops.segment_sum(ll_s, climber_idx, num_segments=n_climbers)
            
        log_lik_climber = jax.vmap(sum_by_climber)(log_lik_obs) # (S, J)
        
        return log_lik_climber

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
