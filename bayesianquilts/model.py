import gzip
import inspect
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import dill
import jax
import jax.numpy as jnp
import numpy as np
from arviz.data import InferenceData
from arviz.data.base import dict_to_dataset
from flax import nnx
from jax import random
from tensorflow_probability.substrates.jax import tf2jax as tf
from tqdm import tqdm

from bayesianquilts.jax.parameter import Interactions
from bayesianquilts.util import training_loop
from bayesianquilts.vi.minibatch import minibatch_fit_surrogate_posterior
import bayesianquilts.tfp_patch

import tensorflow_probability.substrates.jax as tfp
tfmcmc = tfp.mcmc
tfd = tfp.distributions

def FactorizedDistributionMoments(dist, samples=100):
    try:
        # Try analytical moments
        mean = dist.mean()
        var = dist.variance()
        return mean, var
    except Exception:
        # Fallback to sampling
        s = dist.sample(samples)
        if isinstance(s, dict):
            mean = {k: jnp.mean(v, axis=0) for k, v in s.items()}
            var = {k: jnp.var(v, axis=0) for k, v in s.items()}
        else:
            mean = jnp.mean(s, axis=0)
            var = jnp.var(s, axis=0)
        return mean, var


class BayesianModel(nnx.Module, ABC):
    surrogate_distribution: Any = nnx.data(None)
    surrogate_sample: Any = nnx.data(None)
    prior_distribution: Any = nnx.data(None)
    data: Any = nnx.data(None)
    data_cardinality: Any = nnx.data(None)
    params: Any = nnx.data(None)
    var_list: list = []
    bijectors: list = []

    def __init__(
        self,
        data_transform_fn: Callable | None = None,
        dtype: jax.typing.DTypeLike = jnp.float64,
        **kwargs,
    ):
        """Instantiate Model object based on tensorflow dataset
        Arguments:
            data {tf.data or factory} -- Either a dataset or a factory to generate a dataset
        Keyword Arguments:
            data_transform_fn {[type]} -- [description] (default: {None})
            strategy {[type]} -- [description] (default: {None})
        Raises:
            AttributeError: [description]
        """
        super(BayesianModel, self).__init__()
        self.data_transform_fn = data_transform_fn

        self.dtype = dtype

    def fit(
        self,
        batched_data_factory,
        initial_values: Dict[str, jax.typing.ArrayLike] = None,
        **kwargs,
    ):
        def infinite_data_iterator():
            while True:
                # Re-instantiate the iterator from the factory
                iterator = batched_data_factory()
                try:
                    yield from iterator
                except TypeError:
                    # If it's not iterable?
                    yield iterator
        
        res = self._calibrate_minibatch_advi(
            infinite_data_iterator(), initial_values=initial_values, **kwargs
        )
        self.params = res[1]
        return res

    def set_data(self, data, data_transform_fn):
        self.data = data
        self.data_transform_fn = data_transform_fn

    def preprocessor(self):
        return None

    def _calibrate_advi(self):
        pass

    def _calibrate_minibatch_advi(
        self,
        batched_data_factory: Callable,
        batch_size: int,
        dataset_size: int,
        steps_per_epoch: int = 1,
        num_epochs: int = 1,
        accumulation_steps: int = 1,
        check_convergence_every: int = 1,
        sample_size=8,
        sample_batches=1,
        lr_decay_factor: float = 0.5,
        learning_rate=1.0,
        patience: int = 3,
        initial_values: Dict[str, jax.typing.ArrayLike] | None = None,
        unormalized_log_prob_fn: Callable | None = None,
        verbose: bool = True,
        **kwargs,
    ):
        """Calibrate using ADVI

        Args:
            data_factory (callable, required): [description]. Factory
                for generating a batch iterator.
            num_steps (int, optional): [description]. Defaults to 100.
            learning_rate (float, optional): [description]. Defaults to 0.1.
            opt ([type], optional): [description]. Defaults to None.
            abs_tol ([type], optional): [description]. Defaults to 1e-10.
            rel_tol ([type], optional): [description]. Defaults to 1e-8.
            clip_value ([type], optional): [description]. Defaults to 5..
            lr_decay_factor (float, optional): [description]. Defaults to 0.99.
            check_every (int, optional): [description]. Defaults to 25.
            set_expectations (bool, optional): [description]. Defaults to True.
            sample_size (int, optional): [description]. Defaults to 4.
        """

        def run_approximation():
            losses = minibatch_fit_surrogate_posterior(
                target_log_prob_fn=(
                    unormalized_log_prob_fn
                    if unormalized_log_prob_fn is not None
                    else self.unormalized_log_prob
                ),
                initial_values=initial_values,
                surrogate_generator=self.surrogate_distribution_generator,
                surrogate_initializer=self.surrogate_parameter_initializer,
                dataset_size=dataset_size,
                batch_size=batch_size,
                sample_size=sample_size,
                sample_batches=sample_batches,
                learning_rate=learning_rate,
                check_convergence_every=check_convergence_every,
                patience=patience,
                lr_decay_factor=lr_decay_factor,
                steps_per_epoch=steps_per_epoch,
                num_epochs=num_epochs,
                accumulation_steps=accumulation_steps,
                data_iterator=batched_data_factory,
                verbose=verbose,
                **kwargs,
            )
            return losses

        losses, params = run_approximation()
        self.params = params
        return losses, params

    def set_calibration_expectations(self, samples: int = 24, variational: bool = True):
        if variational:
            # Use current params to regenerate surrogate if possible, as self.surrogate_distribution relies on init
            try:
                if hasattr(self, 'surrogate_distribution_generator') and hasattr(self, 'params') and self.params is not None:
                    dist = self.surrogate_distribution_generator(self.params)
                else:
                    dist = self.surrogate_distribution
            except Exception:
                dist = self.surrogate_distribution
                
            mean, var = FactorizedDistributionMoments(
                dist, samples=samples
            )
            self.calibrated_expectations = {k: v for k, v in mean.items()}
            self.calibrated_sd = {k: jnp.sqrt(v) for k, v in var.items()}
        else:
            self.calibrated_expectations = {
                k: jnp.mean(v, axis=0, keepdims=True)
                for k, v in self.surrogate_sample.items()
            }

            self.calibrated_sd = {
                k: jnp.std(v, axis=0, keepdims=True)
                for k, v in self.surrogate_sample.items()
            }

    @abstractmethod
    def predictive_distribution(self, data, **params):
        pass

    def log_likelihood(self, data, **params):
        return self.predictive_distribution(data, return_params=False, **params)[
            "log_likelihood"
        ]

    def sample_stats(
        self, data_factory, params=None, num_samples=100, num_splits=20, data_batches=25
    ):
        likelihood_vars = inspect.getfullargspec(self.log_likelihood).args[1:]

        # split param samples
        params = self.surrogate_sample if params is None else params
        if "data" in likelihood_vars:
            likelihood_vars.remove("data")
        params = (
            self.surrogate_distribution.sample(num_samples)
            if (params is None)
            else params
        )
        if len(likelihood_vars) == 0:
            likelihood_vars = params.keys()
        if "data" in likelihood_vars:
            likelihood_vars.remove("data")
        if len(likelihood_vars) == 0:
            likelihood_vars = self.var_list
        splits = [
            tf.split(value=params[v], num_or_size_splits=num_splits)
            for v in likelihood_vars
        ]

        # reshape the splits
        splits = [
            {k: v for k, v in zip(likelihood_vars, split)} for split in zip(*splits)
        ]
        ll = []
        for batch in tqdm(data_factory()):
            # This should have shape S x N, where S is the number of param
            # samples and N is the batch size
            batch_log_likelihoods = [
                self.log_likelihood(**this_split, data=batch) for this_split in splits
            ]
            batch_log_likelihoods = jnp.concatenate(batch_log_likelihoods, axis=0)
            finite_part = jnp.where(
                jnp.isfinite(batch_log_likelihoods),
                batch_log_likelihoods,
                jnp.zeros_like(batch_log_likelihoods),
            )
            min_val = jnp.min(finite_part)
            #  batch_ll = tf.clip_by_value(batch_ll, min_val-1000, 0.)
            batch_log_likelihoods = jnp.where(
                jnp.finite.is_finite(batch_log_likelihoods),
                batch_log_likelihoods,
                jnp.ones_like(batch_log_likelihoods) * min_val * 1.01,
            )
            ll += [batch_log_likelihoods.numpy()]
        ll = np.concatenate(ll, axis=1)
        ll = np.moveaxis(ll, 0, -1)
        return {
            "log_likelihood": ll.T[np.newaxis, ...],
            "params": {k: v.numpy()[np.newaxis, ...] for k, v in params.items()},
        }

    def save(self, filename="model_save.pkl", gz=True):
        if not isinstance(filename, pathlib.Path):
            filename = pathlib.Path(filename)
        filename = filename.with_suffix(".pkl")
        if not gz:
            with open(filename, "wb") as file:
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)
        else:
            filename = filename.with_suffix(".pkl.gz")
            with gzip.open(filename, "wb") as file:
                #  dill.dump((self.__class__, self), file)
                dill.dump(self, file)


    def unormalized_log_prob_list(self, data, params, prior_weight=1.0):
        dict_params = {k: p for k, p in zip(self.var_list, params)}
        return self.unormalized_log_prob(
            data=data, prior_weight=jnp.astype(prior_weight, self.dtype), **dict_params
        )

    @abstractmethod
    def unormalized_log_prob(
        self,
        data: dict[str, jax.typing.ArrayLike] = None,
        prior_weight: jax.typing.ArrayLike | float = jnp.array(1.0),
        **params,
    ):
        """Generic method for the unormalized log probability function"""
        return

    def fit_projection(
        self, other, data_iterator, nsamples: int = 32, initial_values=None, **kwargs
    ):
        def objective(data, params):
            """Objective function for the training loop."""
            seed = random.PRNGKey(0)
            samples = self.surrogate_distribution_generator(params).sample(
                nsamples, seed=seed
            )
            this_prediction = self.predictive_distribution(data, **samples)[
                "prediction"
            ]
            other_prediction = other.predictive_distribution(data, **other.sample(nsamples))[
                "prediction"
            ]
            delta = other_prediction.kl_divergence(this_prediction)
            return jnp.mean(delta)

        if initial_values is None:
            initial_values = self.params

        return training_loop(
            loss_fn=objective,
            initial_values=initial_values,
            data_iterator=data_iterator,
            **kwargs,
        )

    @abstractmethod
    def create_distributions():
        """Create the prior and surrogate distributions for the model."""
        raise NotImplementedError(
            "This method should be implemented in subclasses to create the "
            "prior and surrogate distributions."
        )


    def transform(self, params):
        return params

    def sample(self, batch_shape=None, prior=False):
        _, sample_key = random.split(random.PRNGKey(np.random.randint(1e8)))
        surrogate = self.surrogate_distribution_generator(self.params)
        if prior:
            if batch_shape is None:
                params = self.prior_distribution.sample(seed=sample_key)
            else:
                params = self.prior_distribution.sample(batch_shape, seed=sample_key)
        elif batch_shape is None:
            return surrogate.sample(seed=sample_key)
        else:
            params = surrogate.sample(batch_shape, seed=sample_key)
        params = self.transform(params)
        return params

    # =========================================================================
    # MCMC Inference Methods
    # =========================================================================

    mcmc_samples: Any = nnx.data(None)  # Storage for MCMC samples

    def fit_mcmc(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        num_chains: int = 4,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        step_size: float = 0.1,
        init_strategy: str = "prior",
        seed: int = None,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """Perform MCMC inference using NUTS (No-U-Turn Sampler).

        Uses Stan-like defaults for robust sampling.

        Args:
            data: Dictionary of data arrays (e.g., {'X': ..., 'y': ...})
            num_chains: Number of independent chains (Stan default: 4)
            num_warmup: Number of warmup/burn-in steps per chain (Stan default: 1000)
            num_samples: Number of post-warmup samples per chain (Stan default: 1000)
            target_accept_prob: Target acceptance probability for dual averaging
                (Stan's adapt_delta, default: 0.8, increase to 0.9-0.99 for difficult posteriors)
            max_tree_depth: Maximum tree depth for NUTS (Stan default: 10)
            step_size: Initial step size for HMC/NUTS (will be adapted during warmup)
            init_strategy: How to initialize chains - "prior" or "zero"
            seed: Random seed (uses random seed if None)
            verbose: Whether to print progress information

        Returns:
            Dictionary mapping parameter names to arrays of shape (num_chains, num_samples, ...)
        """
        if seed is None:
            seed = np.random.randint(0, 2**31)
        key = random.PRNGKey(seed)

        if verbose:
            print(f"Running NUTS with {num_chains} chains...")
            print(f"  Warmup: {num_warmup}, Samples: {num_samples}")
            print(f"  Target acceptance: {target_accept_prob}, Max tree depth: {max_tree_depth}")

        # Create target log probability function
        def target_log_prob_fn(*params_flat):
            params_dict = {k: v for k, v in zip(self.var_list, params_flat)}
            return self.unormalized_log_prob(data=data, prior_weight=1.0, **params_dict)

        # Initialize chains
        key, init_key = random.split(key)
        initial_states = self._create_initial_states(
            num_chains, init_strategy, init_key
        )

        # Run MCMC for each chain
        all_samples = {var: [] for var in self.var_list}
        all_accept_ratios = []

        for chain_idx in range(num_chains):
            key, chain_key = random.split(key)
            
            if verbose:
                print(f"\nChain {chain_idx + 1}/{num_chains}:")

            chain_initial = [initial_states[var][chain_idx] for var in self.var_list]
            
            samples, accept_ratio = self._run_nuts_chain(
                target_log_prob_fn=target_log_prob_fn,
                initial_state=chain_initial,
                num_warmup=num_warmup,
                num_samples=num_samples,
                step_size=step_size,
                target_accept_prob=target_accept_prob,
                max_tree_depth=max_tree_depth,
                seed=chain_key,
                verbose=verbose,
            )

            # Store results if chain is healthy
            if accept_ratio > 1e-6:
                for var, sample in zip(self.var_list, samples):
                    all_samples[var].append(sample)
                all_accept_ratios.append(accept_ratio)
            else:
                if verbose:
                    print(f"  Chain {chain_idx + 1} discarded (acceptance ratio {accept_ratio:.3e} too low)")

        # Stack chains: (num_valid_chains, num_samples, ...)
        result = {}
        num_valid_chains = len(all_accept_ratios)
        
        if num_valid_chains == 0:
            print("WARNING: All chains failed to converge (zero acceptance). Returning last chain to avoid crash.")
            # Fallback: keep the last chain even if bad so we have structure
            for var, sample in zip(self.var_list, samples):
                all_samples[var].append(sample)
            all_accept_ratios.append(accept_ratio)

        for var in self.var_list:
            result[var] = jnp.stack(all_samples[var], axis=0)

        if verbose:
            mean_accept = np.mean(all_accept_ratios)
            print(f"\n--- MCMC Complete ---")
            print(f"Mean acceptance ratio: {mean_accept:.3f}")
            for var in self.var_list:
                # Calculate R-hat
                # potential_scale_reduction expects shape [num_chains, num_samples, ...]
                # but TFP sometimes expects [num_samples, num_chains, ...] depending on version
                # Checking docs: TFP uses [num_samples, num_chains, ...] by default but independent_chain_ndims=1
                # lets us pass [num_chains, num_samples, ...]. 
                # Actually, simpler to just transpose to [num_samples, num_chains, ...]
                samples_transposed = jnp.swapaxes(result[var], 0, 1)
                rhat = tfmcmc.potential_scale_reduction(samples_transposed)
                
                # Check for NaNs in R-hat (can happen if variance is 0)
                rhat = jnp.where(jnp.isnan(rhat), 1.0, rhat)
                max_rhat = jnp.max(rhat)
                
                samples_flat = result[var].reshape(-1, *result[var].shape[2:])
                print(f"  {var}: mean={jnp.mean(samples_flat, axis=0)}, std={jnp.std(samples_flat, axis=0)}, max_rhat={max_rhat:.3f}")

        # Store samples for later use
        self.mcmc_samples = result
        return result

    def _create_initial_states(
        self,
        num_chains: int,
        init_strategy: str,
        key: jax.Array,
    ) -> Dict[str, jnp.ndarray]:
        """Create initial states for MCMC chains.

        Args:
            num_chains: Number of chains
            init_strategy: "prior" to sample from prior, "zero" for zeros
            key: JAX random key

        Returns:
            Dictionary mapping variable names to initial values of shape (num_chains, ...)
        """
        if init_strategy == "prior" and self.prior_distribution is not None:
            # Sample from prior
            samples = self.prior_distribution.sample(num_chains, seed=key)
            if isinstance(samples, dict):
                return samples
            else:
                return {self.var_list[0]: samples}
        elif init_strategy == "zero":
            # Initialize at zero (assumes unconstrained parameterization)
            result = {}
            # Get shapes from surrogate or prior
            if hasattr(self, 'surrogate_parameter_initializer'):
                template = self.surrogate_parameter_initializer(key=key)
                for var in self.var_list:
                    # Find the location parameter for this variable
                    loc_key = f"{var}_loc" if f"{var}_loc" in template else var
                    if loc_key in template:
                        shape = template[loc_key].shape
                        result[var] = jnp.zeros((num_chains,) + shape, dtype=self.dtype)
            return result
        else:
            # Default: small random values
            result = {}
            keys = random.split(key, len(self.var_list))
            if hasattr(self, 'surrogate_parameter_initializer'):
                template = self.surrogate_parameter_initializer(key=key)
                for i, var in enumerate(self.var_list):
                    loc_key = f"{var}_loc" if f"{var}_loc" in template else var
                    if loc_key in template:
                        shape = template[loc_key].shape
                        result[var] = random.normal(keys[i], (num_chains,) + shape, dtype=self.dtype) * 0.1
            return result

    def _run_nuts_chain(
        self,
        target_log_prob_fn: Callable,
        initial_state: list,
        num_warmup: int,
        num_samples: int,
        step_size: float,
        target_accept_prob: float,
        max_tree_depth: int,
        seed: jax.Array,
        verbose: bool = True,
    ) -> tuple:
        """Run a single NUTS chain.

        Args:
            target_log_prob_fn: Unnormalized log probability function
            initial_state: List of initial values for each parameter
            num_warmup: Number of warmup steps
            num_samples: Number of sampling steps
            step_size: Initial step size
            target_accept_prob: Target acceptance probability
            max_tree_depth: Maximum tree depth
            seed: Random key
            verbose: Print progress

        Returns:
            Tuple of (samples_list, mean_accept_ratio)
        """
        # Create NUTS kernel with step size adaptation
        nuts_kernel = tfmcmc.NoUTurnSampler(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
        )

        # Wrap with dual averaging step size adaptation
        adaptive_kernel = tfmcmc.DualAveragingStepSizeAdaptation(
            inner_kernel=nuts_kernel,
            num_adaptation_steps=int(num_warmup * 0.8),  # Adapt for 80% of warmup
            target_accept_prob=target_accept_prob,
        )

        # Run the chain
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
            print("  Running warmup and sampling...")

        samples, is_accepted = run_chain(initial_state, seed)

        # Compute acceptance ratio
        accept_ratio = jnp.mean(is_accepted.astype(jnp.float32))

        if verbose:
            print(f"  Acceptance ratio: {accept_ratio:.3f}")

        return samples, float(accept_ratio)

    def sample_mcmc(self, num_samples: int = None) -> Dict[str, jnp.ndarray]:
        """Sample from stored MCMC results.

        Args:
            num_samples: Number of samples to return. If None, returns all.
                Samples are drawn randomly from all chains.

        Returns:
            Dictionary of parameter samples with shape (num_samples, ...)
        """
        if self.mcmc_samples is None:
            raise ValueError("No MCMC samples available. Run fit_mcmc() first.")

        result = {}
        for var, samples in self.mcmc_samples.items():
            # Flatten chains: (num_chains, num_samples, ...) -> (total, ...)
            flat = samples.reshape(-1, *samples.shape[2:])
            if num_samples is not None and num_samples < flat.shape[0]:
                # Random subsample
                key = random.PRNGKey(np.random.randint(0, 2**31))
                indices = random.choice(key, flat.shape[0], (num_samples,), replace=False)
                result[var] = flat[indices]
            else:
                result[var] = flat
        return result


    def to_arviz(self, data_factory=None):
        sample_stats = self.sample_stats(data_factory=data_factory)
        params = sample_stats["params"]

        idict = {
            "posterior": dict_to_dataset(params),
            "sample_stats": dict_to_dataset(
                {"log_likelihood": sample_stats["log_likelihood"]}
            ),
        }

        return InferenceData(**idict)

class QuiltedBayesianModel(BayesianModel):
    """Quailted Bayesian Model

    Initially a global model, Quilted Bayesian Models can be expanded along a given
    interaction alignment to create a larger model.
    """
    @abstractmethod
    def expand(self, interaction: Interactions):
        pass