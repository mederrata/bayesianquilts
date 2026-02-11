import jax
import jax.numpy as jnp
from typing import Dict, Any
from bayesianquilts.metrics.ais import LikelihoodFunction


class NeuralIRTLikelihood(LikelihoodFunction):
    """
    Likelihood function for the Neural Network IRT Model.

    The model uses a single hidden layer neural network as the item response function:
    - Hidden layer: h = tanh(W1 * theta + b1)
    - Output layer: logits = W2 * h + b2
    - Probabilities: probs = softmax(logits)

    Likelihood is aggregated by person for person-level LOO-IC.

    NOTE: For LOO-IC computation with identity transform only, gradient methods
    return placeholders. For adaptive transforms, the full gradient computation
    would be expensive due to the large number of neural network parameters.
    """

    def __init__(self, dtype=jnp.float64):
        self.dtype = dtype

    def log_likelihood_gradient(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return placeholder gradients for the identity transform case.

        For person-level LOO with identity transform, gradients are not used.
        This placeholder avoids expensive autodiff computation.
        """
        theta = params["theta"]
        S = theta.shape[0]
        N = data["n_people"]

        # Return zeros with appropriate shapes (S, N, param_dim)
        result = {"theta": jnp.zeros((S, N, theta.shape[1]))}

        if "W1" in params:
            W1 = params["W1"]
            result["W1"] = jnp.zeros((S, N, W1.shape[1], W1.shape[2]))
        if "b1" in params:
            b1 = params["b1"]
            result["b1"] = jnp.zeros((S, N, b1.shape[1]))
        if "W2" in params:
            W2 = params["W2"]
            result["W2"] = jnp.zeros((S, N, W2.shape[1], W2.shape[2], W2.shape[3]))
        if "b2" in params:
            b2 = params["b2"]
            result["b2"] = jnp.zeros((S, N, b2.shape[1], b2.shape[2]))

        return result

    def log_likelihood_hessian_diag(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return placeholder Hessian diagonal for the identity transform case.
        """
        # Use same structure as gradient
        return self.log_likelihood_gradient(data, params)

    def _nn_forward(self, theta_val, W1_i, b1_i, W2_i, b2_i):
        """
        Neural network forward pass for a single item.

        Args:
            theta_val: ability value (scalar or array)
            W1_i: input-to-hidden weights (H,)
            b1_i: hidden bias (scalar)
            W2_i: hidden-to-output weights (K, H)
            b2_i: output bias (K,)

        Returns:
            probs: category probabilities (K,) or array with K as last dim
        """
        # Hidden layer: h = tanh(W1 * theta + b1)
        # W1_i: (H,), theta_val: scalar -> h: (H,)
        hidden = jnp.tanh(W1_i * theta_val + b1_i)

        # Output layer: logits = W2 * h + b2
        # W2_i: (K, H), hidden: (H,) -> logits: (K,)
        logits = jnp.dot(W2_i, hidden) + b2_i

        # Softmax for probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        return probs

    def log_likelihood(self, data: Dict[str, Any], params: Dict[str, Any]) -> jnp.ndarray:
        """
        Compute log-likelihood for the Neural IRT Model.

        Parameters (params):
            theta: person abilities (S, J) or (S, N_loo, J) for transformed params

        Data (data) - can also contain frozen params:
            person_idx: indices for each observation (N_obs,)
            item_idx: indices for each observation (N_obs,)
            y: observed responses (0 to K-1) (N_obs,)
            n_people: total number of unique people (J)
            H: number of hidden units
            K: number of response categories
            W1: input-to-hidden weights (S, I, H) - frozen in data
            b1: hidden biases (S, I) - frozen in data
            W2: hidden-to-output weights (S, I, K, H) - frozen in data
            b2: output biases (S, I, K) - frozen in data

        Returns:
            log_lik: logged likelihood aggregated by person
                     (S, J) for original params or (S, N_loo) for transformed params
        """
        person_idx = jnp.asarray(data["person_idx"], dtype=jnp.int32)
        item_idx = jnp.asarray(data["item_idx"], dtype=jnp.int32)
        y = jnp.asarray(data["y"], dtype=jnp.int32)
        n_people = data["n_people"]

        theta = jnp.asarray(params["theta"], dtype=self.dtype)

        # Get neural network weights from params or data (frozen)
        W1 = jnp.asarray(params.get("W1", data.get("W1")), dtype=self.dtype)
        b1 = jnp.asarray(params.get("b1", data.get("b1")), dtype=self.dtype)
        W2 = jnp.asarray(params.get("W2", data.get("W2")), dtype=self.dtype)
        b2 = jnp.asarray(params.get("b2", data.get("b2")), dtype=self.dtype)

        # Check if we have transformed params with extra N_loo dimension
        # Original: theta (S, J)
        # Transformed: theta (S, N_loo, J)
        has_loo_dim = theta.ndim == 3

        if has_loo_dim:
            return self._log_likelihood_transformed(
                theta, W1, b1, W2, b2, person_idx, item_idx, y, n_people
            )
        else:
            return self._log_likelihood_standard(
                theta, W1, b1, W2, b2, person_idx, item_idx, y, n_people
            )

    def _log_likelihood_standard(self, theta, W1, b1, W2, b2, person_idx, item_idx, y, n_people):
        """Compute log-likelihood for standard (non-transformed) params."""
        S = theta.shape[0]
        N_obs = person_idx.shape[0]
        I = W1.shape[1]
        H = W1.shape[2]
        K = W2.shape[2]

        # Gather parameters for each observation
        # theta_obs: (S, N_obs)
        theta_obs = jnp.take_along_axis(theta, person_idx[None, :], axis=1)

        # Gather item parameters for each observation
        # W1_obs: (S, N_obs, H)
        W1_obs = jnp.take_along_axis(W1, item_idx[None, :, None], axis=1)
        # b1_obs: (S, N_obs)
        b1_obs = jnp.take_along_axis(b1, item_idx[None, :], axis=1)
        # W2_obs: (S, N_obs, K, H)
        W2_obs = jnp.take_along_axis(W2, item_idx[None, :, None, None], axis=1)
        # b2_obs: (S, N_obs, K)
        b2_obs = jnp.take_along_axis(b2, item_idx[None, :, None], axis=1)

        # Compute hidden layer: h = tanh(W1 * theta + b1)
        # theta_obs: (S, N_obs) -> (S, N_obs, 1) for broadcasting
        # W1_obs: (S, N_obs, H)
        # hidden: (S, N_obs, H)
        hidden = jnp.tanh(W1_obs * theta_obs[:, :, None] + b1_obs[:, :, None])

        # Output layer: logits = W2 @ h + b2
        # W2_obs: (S, N_obs, K, H), hidden: (S, N_obs, H) -> (S, N_obs, H, 1)
        # logits: (S, N_obs, K)
        logits = jnp.einsum('snkh,snh->snk', W2_obs, hidden) + b2_obs

        # Softmax probabilities
        all_probs = jax.nn.softmax(logits, axis=-1)

        # Select probability for observed y
        # y: (N_obs,) -> need to gather along K dimension
        obs_probs = jnp.take_along_axis(all_probs, y[None, :, None], axis=2)
        obs_probs = jnp.squeeze(obs_probs, axis=2)

        log_lik_obs = jnp.log(jnp.clip(obs_probs, a_min=1e-15))

        # Aggregate by person
        def sum_by_person(ll_s):
            return jax.ops.segment_sum(ll_s, person_idx, num_segments=n_people)

        log_lik_person = jax.vmap(sum_by_person)(log_lik_obs)

        return log_lik_person

    def _log_likelihood_transformed(self, theta, W1, b1, W2, b2, person_idx, item_idx, y, n_people):
        """
        Compute log-likelihood for transformed params with LOO dimension.

        For person-level LOO, when params have shape (S, N_loo, ...), we only need
        the likelihood of person n when using the transformation for LOO case n.
        This returns the diagonal: log_lik[s, n] = likelihood of person n under
        transformed params for leaving out person n.

        theta: (S, N_loo, J)
        W1, b1, W2, b2: (S, I, ...) - typically not transformed, just broadcast

        Returns: (S, N_loo) - log likelihood for each LOO case (diagonal extraction)
        """
        S, N_loo, J = theta.shape
        I = W1.shape[1]
        H = W1.shape[2]
        K = W2.shape[2]
        N_obs = person_idx.shape[0]

        def compute_for_loo_case(theta_s_n, W1_s, b1_s, W2_s, b2_s, loo_idx):
            """Compute likelihood of person loo_idx under params for leaving them out."""
            # theta_s_n: (J,)
            # W1_s: (I, H), b1_s: (I,), W2_s: (I, K, H), b2_s: (I, K)

            # Gather for observations
            theta_obs = theta_s_n[person_idx]  # (N_obs,)
            W1_obs = W1_s[item_idx]  # (N_obs, H)
            b1_obs = b1_s[item_idx]  # (N_obs,)
            W2_obs = W2_s[item_idx]  # (N_obs, K, H)
            b2_obs = b2_s[item_idx]  # (N_obs, K)

            # hidden: (N_obs, H)
            hidden = jnp.tanh(W1_obs * theta_obs[:, None] + b1_obs[:, None])

            # logits: (N_obs, K)
            logits = jnp.einsum('nkh,nh->nk', W2_obs, hidden) + b2_obs

            all_probs = jax.nn.softmax(logits, axis=-1)

            obs_probs = all_probs[jnp.arange(N_obs), y]
            log_lik_obs = jnp.log(jnp.clip(obs_probs, a_min=1e-15))

            # Aggregate by person
            log_lik_person = jax.ops.segment_sum(log_lik_obs, person_idx, num_segments=n_people)

            # Return only the likelihood for the LOO case person
            return log_lik_person[loo_idx]

        loo_indices = jnp.arange(N_loo)

        def compute_all_loo_for_sample(theta_s, W1_s, b1_s, W2_s, b2_s):
            """Compute for all LOO cases within a sample."""
            # theta_s: (N_loo, J)
            # W1_s: (I, H), b1_s: (I,), W2_s: (I, K, H), b2_s: (I, K)
            return jax.vmap(
                lambda t, idx: compute_for_loo_case(t, W1_s, b1_s, W2_s, b2_s, idx)
            )(theta_s, loo_indices)

        # Vmap over samples
        result = jax.vmap(compute_all_loo_for_sample)(theta, W1, b1, W2, b2)
        # result: (S, N_loo)

        return result

    def extract_parameters(self, params: Dict[str, Any]) -> jnp.ndarray:
        """
        Flatten parameters into a single array for AIS.
        Expected keys: theta (S, J), and optionally W1 (S, I, H), b1 (S, I),
                       W2 (S, I, K, H), b2 (S, I, K)
        """
        theta = params["theta"]
        S = theta.shape[0]

        flat_parts = [theta.reshape(S, -1)]

        if "W1" in params:
            flat_parts.append(params["W1"].reshape(S, -1))
        if "b1" in params:
            flat_parts.append(params["b1"].reshape(S, -1))
        if "W2" in params:
            flat_parts.append(params["W2"].reshape(S, -1))
        if "b2" in params:
            flat_parts.append(params["b2"].reshape(S, -1))

        return jnp.concatenate(flat_parts, axis=1)

    def reconstruct_parameters(self, flat_params: jnp.ndarray, template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct parameters from flattened array.
        """
        S = flat_params.shape[0]
        result = {}
        idx = 0

        # theta
        J = template["theta"].shape[1]
        result["theta"] = flat_params[:, idx:idx+J].reshape(S, J)
        idx += J

        # Optional neural network params
        if "W1" in template:
            I, H = template["W1"].shape[1], template["W1"].shape[2]
            size = I * H
            result["W1"] = flat_params[:, idx:idx+size].reshape(S, I, H)
            idx += size

        if "b1" in template:
            I = template["b1"].shape[1]
            result["b1"] = flat_params[:, idx:idx+I].reshape(S, I)
            idx += I

        if "W2" in template:
            I, K, H = template["W2"].shape[1], template["W2"].shape[2], template["W2"].shape[3]
            size = I * K * H
            result["W2"] = flat_params[:, idx:idx+size].reshape(S, I, K, H)
            idx += size

        if "b2" in template:
            I, K = template["b2"].shape[1], template["b2"].shape[2]
            size = I * K
            result["b2"] = flat_params[:, idx:idx+size].reshape(S, I, K)
            idx += size

        return result
