#!/usr/bin/env python3
"""Frequentist model classes for bayesianquilts.

Provides FrequentistModel and QuiltedFrequentistModel as frequentist
counterparts to BayesianModel and QuiltedBayesianModel, supporting
minibatch training via gradient descent with L1/L2 regularization.

These classes implement the associated MAP (maximum a posteriori) estimation
problem rather than full Bayesian inference. While the WAIC analysis in the
manuscript is most naturally Bayesian, the model fits use MAP estimation
because full Bayesian inference (MCMC or variational) has significant
computational overhead that is impractical for large-scale benchmarking.

The regularization framework implements theory from:
    Chang (2025), "A renormalization-group inspired hierarchical Bayesian
    framework for piecewise linear regression models"

Key theory-based features:
    - Generalization-preserving regularization: τ ≤ σ/√(p·N)
    - Order-dependent L1 scaling by interaction complexity
    - Effective degrees of freedom bound: df_eff = tr[(X'X + τ⁻²I)⁻¹X'X]
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from bayesianquilts.jax.parameter import Decomposed, Interactions, MultiwayContingencyTable


class FrequentistModel(nnx.Module, ABC):
    """Base class for frequentist models with minibatch training support.

    Implements MAP (maximum a posteriori) estimation via gradient descent
    with L1/L2 regularization. This is the frequentist analog of BayesianModel,
    used when full Bayesian inference is computationally prohibitive.

    Provides fit() for minibatch gradient descent and fit_full() for
    full-batch training. Subclasses must implement:
    - loss(data, params) -> scalar loss (negative log-likelihood)
    - predict(data, params) -> predictions

    The objective function optimized is:
        argmin_θ { -log p(y|X,θ) + λ₂||θ||² + λ₁||θ||₁ }

    which corresponds to MAP estimation under a Gaussian/Laplace prior.
    """

    params: Optional[Dict[str, Any]] = nnx.data(None)
    training_history: Optional[Dict[str, List]] = nnx.data(None)
    noise_scale_estimate: Optional[float] = nnx.data(None)

    def __init__(
        self,
        dtype: jax.typing.DTypeLike = jnp.float32,
        **kwargs,
    ):
        """Initialize FrequentistModel.

        Args:
            dtype: Data type for parameters
            **kwargs: Additional arguments
        """
        super().__init__()
        self.dtype = dtype
        self.params = None
        self.training_history = None
        self.noise_scale_estimate = None

    @abstractmethod
    def loss(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Dict[str, jax.typing.ArrayLike],
    ) -> jax.typing.ArrayLike:
        """Compute loss for a batch of data.

        Args:
            data: Batch data dictionary
            params: Model parameters

        Returns:
            Scalar loss value
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Optional[Dict[str, jax.typing.ArrayLike]] = None,
    ) -> jax.typing.ArrayLike:
        """Generate predictions for data.

        Args:
            data: Data dictionary
            params: Model parameters (uses self.params if None)

        Returns:
            Predictions array
        """
        pass

    def _initialize_params(self) -> Dict[str, jax.typing.ArrayLike]:
        """Initialize parameters. Override in subclasses."""
        raise NotImplementedError(
            "Subclass must implement _initialize_params or provide initial_values to fit()"
        )

    def _get_param_order(self, param_name: str) -> int:
        """Get interaction order of a parameter. Override in subclasses."""
        return 0

    def estimate_noise_scale(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Optional[Dict[str, jax.typing.ArrayLike]] = None,
    ) -> float:
        """Estimate noise scale from residuals.

        For regression, computes σ = √(RSS / (n - p)).
        For classification, returns 1.0 (not applicable).
        Override in subclasses for specific implementations.

        Args:
            data: Dataset with observations
            params: Fitted parameters (uses self.params if None)

        Returns:
            Estimated noise standard deviation σ̂
        """
        return 1.0

    def effective_degrees_of_freedom(
        self,
        X: jax.typing.ArrayLike,
        regularization_strength: float,
    ) -> float:
        """Compute effective degrees of freedom.

        df_eff = tr[(X'X + τ⁻²I)⁻¹X'X]

        This is the leverage-based formula for ridge regression.
        For other models, this provides an approximation.

        Args:
            X: Design matrix (n_samples, n_features)
            regularization_strength: τ (prior standard deviation)

        Returns:
            Effective degrees of freedom
        """
        X = jnp.asarray(X)
        n, p = X.shape
        XtX = X.T @ X
        tau_inv_sq = 1.0 / (regularization_strength ** 2 + 1e-10)
        regularized = XtX + tau_inv_sq * jnp.eye(p)
        inv_reg = jnp.linalg.inv(regularized)
        df_eff = jnp.trace(inv_reg @ XtX)
        return float(df_eff)

    def fit(
        self,
        batched_data_factory: Callable,
        initial_values: Optional[Dict[str, jax.typing.ArrayLike]] = None,
        num_epochs: int = 100,
        steps_per_epoch: int = 100,
        learning_rate: float = 0.01,
        optimizer: Optional[optax.GradientTransformation] = None,
        clip_norm: Optional[float] = 1.0,
        l1_weight: float = 0.0,
        l2_weight: float = 0.01,
        order_dependent_l1: bool = False,
        elastic_net_alpha: Optional[float] = None,
        patience: Optional[int] = None,
        lr_decay_factor: float = 0.5,
        verbose: bool = True,
        eval_data: Optional[Dict] = None,
        eval_every: int = 10,
    ) -> Tuple[List[float], Dict[str, jax.typing.ArrayLike]]:
        """Fit model using minibatch gradient descent.

        Implements MAP estimation with regularization. The objective is:
            argmin_θ { L(θ) + λ₂||θ||² + λ₁||θ||₁ }

        Args:
            batched_data_factory: Callable that returns an iterator over batches
            initial_values: Initial parameter values (None = auto-initialize)
            num_epochs: Number of training epochs
            steps_per_epoch: Gradient steps per epoch
            learning_rate: Initial learning rate
            optimizer: Custom optax optimizer (overrides learning_rate/clip_norm)
            clip_norm: Gradient clipping threshold (None to disable)
            l1_weight: L1 regularization weight (Laplace prior)
            l2_weight: L2 regularization weight (Gaussian prior)
            order_dependent_l1: If True, scale L1 by interaction order (k+1)
            elastic_net_alpha: If set, use elastic net mixing (alpha=1 pure L1)
            patience: Early stopping patience (None to disable)
            lr_decay_factor: LR decay factor on plateau
            verbose: Print progress
            eval_data: Optional held-out data for validation
            eval_every: Evaluate on eval_data every N epochs

        Returns:
            Tuple of (loss_history, final_params)
        """
        if initial_values is None:
            initial_values = self._initialize_params()

        params = {k: jnp.array(v, dtype=self.dtype) for k, v in initial_values.items()}

        if optimizer is None:
            opt_chain = [optax.adam(learning_rate)]
            if clip_norm is not None:
                opt_chain.insert(0, optax.clip_by_global_norm(clip_norm))
            optimizer = optax.chain(*opt_chain)

        opt_state = optimizer.init(params)

        _l1 = l1_weight
        _l2 = l2_weight
        if elastic_net_alpha is not None:
            _l1 = elastic_net_alpha * (l1_weight + l2_weight)
            _l2 = (1 - elastic_net_alpha) * (l1_weight + l2_weight)

        def regularized_loss(data, params):
            base_loss = self.loss(data, params)
            reg = jnp.array(0.0, dtype=self.dtype)
            for name, val in params.items():
                if _l2 > 0:
                    reg = reg + _l2 * jnp.sum(val ** 2)
                if _l1 > 0:
                    if order_dependent_l1:
                        order = self._get_param_order(name)
                        reg = reg + _l1 * (order + 1) * jnp.sum(jnp.abs(val))
                    else:
                        reg = reg + _l1 * jnp.sum(jnp.abs(val))
            return base_loss + reg

        @jax.jit
        def step(params, opt_state, batch):
            loss_val, grads = jax.value_and_grad(regularized_loss, argnums=1)(batch, params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_val

        loss_history = []
        eval_history = []
        best_loss = float('inf')
        best_params = params
        patience_counter = 0
        current_lr = learning_rate

        def data_iterator():
            while True:
                iterator = batched_data_factory()
                try:
                    yield from iterator
                except TypeError:
                    yield iterator

        data_iter = data_iterator()

        epoch_iter = tqdm(range(num_epochs), desc="Training") if verbose else range(num_epochs)

        for epoch in epoch_iter:
            epoch_losses = []

            for _ in range(steps_per_epoch):
                batch = next(data_iter)
                params, opt_state, loss_val = step(params, opt_state, batch)
                epoch_losses.append(float(loss_val))

            epoch_loss = np.mean(epoch_losses)
            loss_history.append(epoch_loss)

            if eval_data is not None and (epoch + 1) % eval_every == 0:
                eval_loss = float(self.loss(eval_data, params))
                eval_history.append(eval_loss)
                if verbose:
                    tqdm.write(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}, eval_loss={eval_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_params = {k: v.copy() for k, v in params.items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience is not None and patience_counter >= patience:
                current_lr *= lr_decay_factor
                if current_lr < 1e-7:
                    if verbose:
                        tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
                opt_chain = [optax.adam(current_lr)]
                if clip_norm is not None:
                    opt_chain.insert(0, optax.clip_by_global_norm(clip_norm))
                optimizer = optax.chain(*opt_chain)
                opt_state = optimizer.init(params)
                patience_counter = 0
                if verbose:
                    tqdm.write(f"Reducing LR to {current_lr:.2e}")

        self.params = best_params
        self.training_history = {
            'train_loss': loss_history,
            'eval_loss': eval_history if eval_data is not None else None,
        }

        return loss_history, best_params

    def fit_full(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        initial_values: Optional[Dict[str, jax.typing.ArrayLike]] = None,
        num_epochs: int = 1000,
        learning_rate: float = 0.01,
        **kwargs,
    ) -> Tuple[List[float], Dict[str, jax.typing.ArrayLike]]:
        """Fit model using full-batch gradient descent.

        Convenience wrapper around fit() for non-minibatch training.

        Args:
            data: Full dataset dictionary
            initial_values: Initial parameter values
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            **kwargs: Additional arguments passed to fit()

        Returns:
            Tuple of (loss_history, final_params)
        """
        def data_factory():
            return iter([data])

        return self.fit(
            batched_data_factory=data_factory,
            initial_values=initial_values,
            num_epochs=num_epochs,
            steps_per_epoch=1,
            learning_rate=learning_rate,
            **kwargs,
        )


class QuiltedFrequentistModel(FrequentistModel):
    """Frequentist model with hierarchical parameter decomposition.

    Supports the same decomposition structure as QuiltedBayesianModel but
    uses frequentist estimation via gradient descent with regularization.

    Provides:
    - Automatic parameter initialization from decompositions
    - Order-dependent regularization
    - Generalization-preserving regularization scaling
    - Staged fitting by interaction order
    """

    def __init__(
        self,
        dtype: jax.typing.DTypeLike = jnp.float32,
        noise_scale: float = 1.0,
        total_n: Optional[int] = None,
        use_generalization_preserving: bool = True,
        df_eff_bound: float = 0.5,
        per_component_bound: bool = True,
        **kwargs,
    ):
        """Initialize QuiltedFrequentistModel.

        Args:
            dtype: Data type for parameters
            noise_scale: Estimated noise scale (for regularization scaling)
            total_n: Total sample size (for regularization scaling)
            use_generalization_preserving: Use theory-derived regularization
            df_eff_bound: Target effective df bound per component
            per_component_bound: Bound df per component vs per parameter
            **kwargs: Additional arguments
        """
        super().__init__(dtype=dtype, **kwargs)
        self.noise_scale = noise_scale
        self.total_n = total_n
        self.use_generalization_preserving = use_generalization_preserving
        self.df_eff_bound = df_eff_bound
        self.per_component_bound = per_component_bound

    def _get_decompositions(self) -> Dict[str, Decomposed]:
        """Return all Decomposed objects in this model."""
        decomps = {}
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
            try:
                val = getattr(self, attr_name)
            except Exception:
                continue
            if isinstance(val, Decomposed):
                decomps[attr_name] = val
        return decomps

    def _initialize_params(self) -> Dict[str, jax.typing.ArrayLike]:
        """Initialize parameters from decompositions."""
        decomps = self._get_decompositions()
        if not decomps:
            raise ValueError("No decompositions found. Override _initialize_params.")

        params = {}
        for decomp in decomps.values():
            tensors, _, _ = decomp.generate_tensors(dtype=self.dtype)
            for name, tensor in tensors.items():
                params[name] = jnp.zeros_like(tensor)

        return params

    def _get_regularization_scales(self) -> Dict[str, float]:
        """Get per-component regularization scales.

        Uses generalization-preserving scaling if enabled and total_n is set.
        """
        decomps = self._get_decompositions()
        scales = {}

        for decomp in decomps.values():
            if self.use_generalization_preserving and self.total_n is not None:
                comp_scales = decomp.generalization_preserving_scales(
                    noise_scale=self.noise_scale,
                    total_n=self.total_n,
                    c=self.df_eff_bound,
                    per_component=self.per_component_bound,
                )
                scales.update(comp_scales)
            else:
                for name in decomp._tensor_parts.keys():
                    order = decomp.component_order(name)
                    scales[name] = 1.0 / (2 ** order)

        return scales

    def _get_param_order(self, param_name: str) -> int:
        """Get interaction order of a parameter."""
        decomps = self._get_decompositions()
        for decomp in decomps.values():
            if param_name in decomp._tensor_parts:
                return decomp.component_order(param_name)
        return 0

    def regularized_loss(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Dict[str, jax.typing.ArrayLike],
        l2_weight: float = 0.01,
        order_dependent: bool = True,
    ) -> jax.typing.ArrayLike:
        """Compute loss with order-dependent regularization.

        Args:
            data: Batch data
            params: Model parameters
            l2_weight: Base L2 regularization weight
            order_dependent: Scale regularization by interaction order

        Returns:
            Regularized loss value
        """
        base_loss = self.loss(data, params)

        if l2_weight <= 0:
            return base_loss

        scales = self._get_regularization_scales() if order_dependent else {}

        reg = jnp.array(0.0, dtype=self.dtype)
        for name, val in params.items():
            scale = scales.get(name, 1.0)
            reg = reg + l2_weight * jnp.sum(val ** 2) / (scale ** 2 + 1e-8)

        return base_loss + reg

    def fit(
        self,
        batched_data_factory: Callable,
        initial_values: Optional[Dict[str, jax.typing.ArrayLike]] = None,
        l2_weight: float = 0.01,
        order_dependent_reg: bool = True,
        **kwargs,
    ) -> Tuple[List[float], Dict[str, jax.typing.ArrayLike]]:
        """Fit model with order-dependent regularization.

        Wraps parent fit() to use regularized_loss with decomposition-aware
        regularization scaling.

        Args:
            batched_data_factory: Callable returning batch iterator
            initial_values: Initial parameter values
            l2_weight: Base L2 regularization weight
            order_dependent_reg: Use order-dependent regularization scaling
            **kwargs: Additional arguments for parent fit()

        Returns:
            Tuple of (loss_history, final_params)
        """
        if initial_values is None:
            initial_values = self._initialize_params()

        original_loss = self.loss

        def wrapped_loss(data, params):
            return self.regularized_loss(
                data, params,
                l2_weight=l2_weight,
                order_dependent=order_dependent_reg,
            )

        self.loss = wrapped_loss

        try:
            result = super().fit(
                batched_data_factory=batched_data_factory,
                initial_values=initial_values,
                l2_weight=0.0,
                **kwargs,
            )
        finally:
            self.loss = original_loss

        return result

    def staged_fit(
        self,
        batched_data_factory: Callable,
        max_order: Optional[int] = None,
        epochs_per_stage: int = 50,
        final_epochs: int = 100,
        sparsity_threshold: float = 0.1,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[List[float], Dict[str, jax.typing.ArrayLike]]:
        """Train progressively by interaction order.

        For each order 0..max_order:
        1. Initialize new components at this order
        2. Train all components up to this order
        3. Assess sparsity, optionally prune sparse components

        Args:
            batched_data_factory: Callable returning batch iterator
            max_order: Maximum interaction order (None = auto-detect)
            epochs_per_stage: Epochs per stage
            final_epochs: Epochs for final joint training
            sparsity_threshold: Threshold for considering component sparse
            verbose: Print progress
            **kwargs: Additional arguments for fit()

        Returns:
            Tuple of (loss_history, final_params)
        """
        decomps = self._get_decompositions()

        if not decomps:
            return self.fit(batched_data_factory, **kwargs)

        if max_order is None:
            max_order = max(d.max_order() for d in decomps.values())

        all_losses = []
        params = self._initialize_params()

        for order in range(max_order + 1):
            if verbose:
                print(f"\n=== Stage {order}: Training up to order {order} ===")

            active_params = {}
            for decomp in decomps.values():
                for name in decomp._tensor_parts.keys():
                    if decomp.component_order(name) <= order:
                        active_params[name] = params[name]

            original_loss = self.loss

            def staged_loss(data, active_p):
                full_params = {**params}
                full_params.update(active_p)
                return original_loss(data, full_params)

            self.loss = staged_loss

            try:
                stage_losses, stage_params = super().fit(
                    batched_data_factory=batched_data_factory,
                    initial_values=active_params,
                    num_epochs=epochs_per_stage if order < max_order else final_epochs,
                    verbose=verbose,
                    **kwargs,
                )
            finally:
                self.loss = original_loss

            params.update(stage_params)
            all_losses.extend(stage_losses)

            if verbose:
                sparse = self._assess_sparsity(params, threshold=sparsity_threshold)
                if sparse:
                    print(f"  Sparse components: {sparse}")

        self.params = params
        return all_losses, params

    def _assess_sparsity(
        self,
        params: Dict[str, jax.typing.ArrayLike],
        threshold: float = 0.1,
    ) -> List[str]:
        """Identify sparse (near-zero) components.

        Args:
            params: Current parameters
            threshold: Relative norm threshold for sparsity

        Returns:
            List of sparse component names
        """
        decomps = self._get_decompositions()
        sparse = []

        max_norm = max(float(jnp.linalg.norm(v)) for v in params.values())
        if max_norm < 1e-10:
            return []

        for name, val in params.items():
            for decomp in decomps.values():
                if name in decomp._tensor_parts:
                    rel_norm = float(jnp.linalg.norm(val)) / max_norm
                    if rel_norm < threshold:
                        sparse.append(name)
                    break

        return sparse

    def set_contingency_table(
        self,
        contingency_table: MultiwayContingencyTable,
    ) -> None:
        """Set contingency table for exact sample size computation.

        When set, generalization_preserving_scales() will use exact local
        sample sizes from the contingency table rather than assuming uniform
        distribution.

        Args:
            contingency_table: Fitted MultiwayContingencyTable
        """
        self._contingency_table = contingency_table

    def _get_regularization_scales_with_counts(self) -> Dict[str, float]:
        """Get regularization scales using contingency table if available."""
        decomps = self._get_decompositions()
        scales = {}
        contingency = getattr(self, '_contingency_table', None)

        for decomp in decomps.values():
            if self.use_generalization_preserving:
                comp_scales = decomp.generalization_preserving_scales(
                    noise_scale=self.noise_scale,
                    total_n=self.total_n,
                    contingency_table=contingency,
                    c=self.df_eff_bound,
                    per_component=self.per_component_bound,
                )
                scales.update(comp_scales)
            else:
                for name in decomp._tensor_parts.keys():
                    order = decomp.component_order(name)
                    scales[name] = 1.0 / (2 ** order)

        return scales

    def component_df_eff(
        self,
        params: Dict[str, jax.typing.ArrayLike],
    ) -> Dict[str, float]:
        """Compute effective degrees of freedom per component.

        Uses the approximation df_eff ≈ p · τ²/(τ² + σ²/N) for each component,
        where p is the number of parameters, τ is the prior scale, and N is
        the local sample size.

        Args:
            params: Current parameter values

        Returns:
            Dictionary mapping component names to effective df
        """
        decomps = self._get_decompositions()
        scales = self._get_regularization_scales_with_counts()
        df_effs = {}

        contingency = getattr(self, '_contingency_table', None)

        for decomp in decomps.values():
            for name, shape in decomp._tensor_part_shapes.items():
                tau = scales.get(name, 1.0)
                p = int(np.prod(shape))
                interaction_vars = decomp._tensor_part_interactions[name]
                interaction_shape = shape[: -len(decomp._param_shape)] if decomp._param_shape else shape

                if contingency is not None and len(interaction_vars) > 0:
                    counts = contingency.lookup(interaction_vars)
                    n_local = float(np.mean(counts))
                elif self.total_n is not None:
                    n_cells = int(np.prod(interaction_shape))
                    n_local = self.total_n / max(n_cells, 1)
                else:
                    n_local = 1.0

                sigma_sq_over_n = (self.noise_scale ** 2) / max(n_local, 1)
                tau_sq = tau ** 2
                df_eff = p * tau_sq / (tau_sq + sigma_sq_over_n)
                df_effs[name] = float(df_eff)

        return df_effs

    def total_df_eff(
        self,
        params: Dict[str, jax.typing.ArrayLike],
    ) -> float:
        """Compute total effective degrees of freedom.

        Args:
            params: Current parameter values

        Returns:
            Total effective df across all components
        """
        return sum(self.component_df_eff(params).values())

    def generalization_gap(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Dict[str, jax.typing.ArrayLike],
        max_order: int,
        cross_validate: bool = False,
        n_folds: int = 5,
    ) -> Dict[int, float]:
        """Compute generalization gap ΔS_K for each truncation order K.

        The generalization gap measures the change in expected predictive
        accuracy from truncating at order K. Negative values indicate
        improved generalization; positive values indicate overfitting.

        This is the MAP analog of the Bayesian WAIC analysis from the paper.
        Since we're doing MAP estimation, we use cross-validation or AIC-like
        approximations rather than full WAIC.

        Args:
            data: Full dataset
            params: Fitted parameters (at max_order)
            max_order: Maximum order included in params
            cross_validate: If True, use cross-validation (slower but accurate)
            n_folds: Number of CV folds

        Returns:
            Dict mapping order K to generalization gap ΔS_K
        """
        decomps = self._get_decompositions()
        gaps = {}

        base_loss = float(self.loss(data, params))
        base_df = self.total_df_eff(params)

        for K in range(max_order + 1):
            truncated_params = {}
            for decomp in decomps.values():
                for name in decomp._tensor_parts.keys():
                    if decomp.component_order(name) <= K:
                        truncated_params[name] = params.get(name, jnp.zeros_like(decomp._tensor_parts[name]))
                    else:
                        truncated_params[name] = jnp.zeros_like(decomp._tensor_parts[name])

            trunc_loss = float(self.loss(data, truncated_params))
            trunc_df = sum(
                v for k, v in self.component_df_eff(truncated_params).items()
                if any(decomp.component_order(k) <= K for decomp in decomps.values() if k in decomp._tensor_parts)
            )

            if data.get('N') is not None:
                N = data['N']
            elif 'X' in data:
                N = len(data['X'])
            elif 'y' in data:
                N = len(data['y'])
            else:
                N = self.total_n or 1

            aic_base = base_loss + 2 * base_df / N
            aic_trunc = trunc_loss + 2 * trunc_df / N
            gaps[K] = aic_trunc - aic_base

        return gaps

    def optimal_truncation_order(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Dict[str, jax.typing.ArrayLike],
        max_order: int,
    ) -> int:
        """Find optimal truncation order via RG flow analysis.

        Implements the renormalization group flow selection from the paper:
        truncate at the smallest K where ΔS_K ≤ 0 (no generalization loss).

        Args:
            data: Dataset
            params: Fitted parameters at max_order
            max_order: Maximum order in params

        Returns:
            Optimal truncation order K*
        """
        gaps = self.generalization_gap(data, params, max_order)

        for K in range(max_order + 1):
            if gaps[K] <= 0:
                return K

        return max_order

    def estimate_noise_scale_logistic(
        self,
        y: jax.typing.ArrayLike,
        p_hat: jax.typing.ArrayLike,
    ) -> float:
        """Estimate effective noise scale for logistic regression.

        For logistic regression, the Fisher information for observation i is
        I_i = p_i(1 - p_i). The effective noise variance is the inverse of
        the average Fisher information:

            σ²_eff = 1 / E[p(1-p)]

        This yields σ_eff for use in generalization-preserving regularization.

        Reference:
            Chang (2025), Section 3.4: "Extension to GLMs"

        Args:
            y: Binary outcomes (0/1), shape (n,)
            p_hat: Predicted probabilities, shape (n,)

        Returns:
            Effective noise scale σ_eff for regularization scaling
        """
        p_hat = jnp.clip(jnp.asarray(p_hat), 1e-6, 1 - 1e-6)
        fisher_weights = p_hat * (1 - p_hat)
        avg_fisher = jnp.mean(fisher_weights)
        sigma_eff = 1.0 / jnp.sqrt(avg_fisher)
        return float(sigma_eff)

    def update_regularization_from_predictions(
        self,
        y: jax.typing.ArrayLike,
        p_hat: jax.typing.ArrayLike,
    ) -> None:
        """Update regularization scales using current predictions.

        Computes σ_eff from Fisher information and updates the noise_scale
        used in generalization-preserving regularization.

        Call this after an initial fit to get theory-based regularization.

        Args:
            y: Binary outcomes
            p_hat: Current predicted probabilities
        """
        self.noise_scale = self.estimate_noise_scale_logistic(y, p_hat)
        self.noise_scale_estimate = self.noise_scale

    def fit_with_model_selection(
        self,
        batched_data_factory: Callable,
        validation_data: Optional[Dict[str, jax.typing.ArrayLike]] = None,
        max_order: Optional[int] = None,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[List[float], Dict[str, jax.typing.ArrayLike], int]:
        """Fit with automatic model selection via RG flow.

        Trains the full model, then determines optimal truncation order
        using generalization gap analysis.

        Args:
            batched_data_factory: Callable returning batch iterator
            validation_data: Held-out data for model selection
            max_order: Maximum interaction order (None = auto-detect)
            verbose: Print progress
            **kwargs: Additional arguments for fit()

        Returns:
            Tuple of (losses, params, optimal_order)
        """
        losses, params = self.staged_fit(
            batched_data_factory=batched_data_factory,
            max_order=max_order,
            verbose=verbose,
            **kwargs,
        )

        decomps = self._get_decompositions()
        if not decomps:
            return losses, params, 0

        actual_max_order = max(d.max_order() for d in decomps.values())
        if max_order is not None:
            actual_max_order = min(actual_max_order, max_order)

        if validation_data is not None:
            eval_data = validation_data
        else:
            batch = next(iter(batched_data_factory()))
            eval_data = batch

        K_star = self.optimal_truncation_order(eval_data, params, actual_max_order)

        if verbose:
            gaps = self.generalization_gap(eval_data, params, actual_max_order)
            print(f"\nGeneralization gaps by order:")
            for K, gap in gaps.items():
                marker = " <-- optimal" if K == K_star else ""
                print(f"  K={K}: ΔS = {gap:+.4f}{marker}")

        for decomp in decomps.values():
            for name in decomp._tensor_parts.keys():
                if decomp.component_order(name) > K_star:
                    params[name] = jnp.zeros_like(params[name])

        self.params = params
        return losses, params, K_star


class LogisticQuiltedModel(QuiltedFrequentistModel):
    """Quilted logistic regression with theory-based regularization.

    Implements hierarchical logistic regression with automatic Fisher
    information-based noise scale estimation for theory-derived
    regularization scaling.

    The regularization follows the generalization-preserving bound:
        τ^(α) = σ_eff / √(p · N^(α))

    where σ_eff = 1/√E[p(1-p)] is computed from the Fisher information.

    Example usage:
        >>> from bayesianquilts.jax.parameter import Decomposed, Interactions
        >>> interactions = Interactions([('x1', 4), ('x2', 4)])
        >>> model = LogisticQuiltedModel(
        ...     interactions=interactions,
        ...     n_features=10,
        ...     total_n=1000,
        ... )
        >>> losses, params = model.fit_logistic(
        ...     X_train, y_train,
        ...     num_epochs=100,
        ... )
        >>> probs = model.predict_proba(X_test)
    """

    def __init__(
        self,
        interactions: "Interactions",
        n_features: int,
        total_n: int,
        max_order: Optional[int] = None,
        dtype: jax.typing.DTypeLike = jnp.float32,
        use_generalization_preserving: bool = True,
        df_eff_bound: float = 0.5,
        per_component_bound: bool = True,
        **kwargs,
    ):
        """Initialize LogisticQuiltedModel.

        Args:
            interactions: Interactions object defining lattice structure
            n_features: Number of input features
            total_n: Total training sample size
            max_order: Maximum interaction order (None = use all)
            dtype: Data type for parameters
            use_generalization_preserving: Use theory-based regularization
            df_eff_bound: Target df_eff bound per component
            per_component_bound: Bound df per component vs per parameter
        """
        super().__init__(
            dtype=dtype,
            noise_scale=2.0,  # Initial estimate; updated during fit
            total_n=total_n,
            use_generalization_preserving=use_generalization_preserving,
            df_eff_bound=df_eff_bound,
            per_component_bound=per_component_bound,
            **kwargs,
        )
        self.n_features = n_features
        self._max_order = max_order

        # Create decompositions for intercept and regression coefficients
        self.intercept_decomp = Decomposed(
            interactions=interactions,
            param_shape=[1],
            name="intercept",
            dtype=dtype,
            max_order=max_order,
        )
        self.beta_decomp = Decomposed(
            interactions=interactions,
            param_shape=[n_features],
            name="beta",
            dtype=dtype,
            max_order=max_order,
        )

    def loss(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Dict[str, jax.typing.ArrayLike],
    ) -> jax.typing.ArrayLike:
        """Compute negative log-likelihood for logistic regression.

        Args:
            data: Dict with 'X' (features), 'y' (labels), 'indices' (lattice indices)
            params: Model parameters

        Returns:
            Negative log-likelihood (scalar)
        """
        X = jnp.asarray(data['X'], dtype=self.dtype)
        y = jnp.asarray(data['y'], dtype=self.dtype)
        indices = jnp.asarray(data['indices'], dtype=jnp.int32)

        intercept_params = {k: v for k, v in params.items() if k.startswith('intercept')}
        beta_params = {k: v for k, v in params.items() if k.startswith('beta')}

        intercept = self.intercept_decomp.lookup_flat(indices, intercept_params, dtype=self.dtype)
        intercept = intercept[..., 0]  # Remove param dim

        beta = self.beta_decomp.lookup_flat(indices, beta_params, dtype=self.dtype)

        logits = jnp.sum(X * beta, axis=-1) + intercept
        nll = -jnp.mean(y * logits - jnp.logaddexp(0, logits))
        return nll

    def predict(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Optional[Dict[str, jax.typing.ArrayLike]] = None,
    ) -> jax.typing.ArrayLike:
        """Predict logits.

        Args:
            data: Dict with 'X' and 'indices'
            params: Model parameters (uses self.params if None)

        Returns:
            Logits array
        """
        if params is None:
            params = self.params

        X = jnp.asarray(data['X'], dtype=self.dtype)
        indices = jnp.asarray(data['indices'], dtype=jnp.int32)

        intercept_params = {k: v for k, v in params.items() if k.startswith('intercept')}
        beta_params = {k: v for k, v in params.items() if k.startswith('beta')}

        intercept = self.intercept_decomp.lookup_flat(indices, intercept_params, dtype=self.dtype)
        intercept = intercept[..., 0]

        beta = self.beta_decomp.lookup_flat(indices, beta_params, dtype=self.dtype)

        logits = jnp.sum(X * beta, axis=-1) + intercept
        return logits

    def predict_proba(
        self,
        data: Dict[str, jax.typing.ArrayLike],
        params: Optional[Dict[str, jax.typing.ArrayLike]] = None,
    ) -> jax.typing.ArrayLike:
        """Predict probabilities.

        Args:
            data: Dict with 'X' and 'indices'
            params: Model parameters (uses self.params if None)

        Returns:
            Probability array
        """
        logits = self.predict(data, params)
        return jax.nn.sigmoid(logits)

    def fit_logistic(
        self,
        X: jax.typing.ArrayLike,
        y: jax.typing.ArrayLike,
        indices: jax.typing.ArrayLike,
        num_epochs: int = 100,
        learning_rate: float = 0.01,
        batch_size: Optional[int] = None,
        warmup_epochs: int = 20,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[List[float], Dict[str, jax.typing.ArrayLike]]:
        """Fit logistic regression with automatic regularization scaling.

        Two-phase fitting:
        1. Warmup: Fit with default regularization to get initial predictions
        2. Main: Update σ_eff from Fisher information, refit with theory scaling

        Args:
            X: Feature matrix, shape (n_samples, n_features)
            y: Binary labels, shape (n_samples,)
            indices: Lattice indices, shape (n_samples, n_dims)
            num_epochs: Total training epochs
            learning_rate: Learning rate
            batch_size: Minibatch size (None for full batch)
            warmup_epochs: Epochs for initial fit before updating regularization
            verbose: Print progress
            **kwargs: Additional arguments for fit()

        Returns:
            Tuple of (loss_history, final_params)
        """
        X = jnp.asarray(X, dtype=self.dtype)
        y = jnp.asarray(y, dtype=self.dtype)
        indices = jnp.asarray(indices, dtype=jnp.int32)
        n_samples = len(y)

        if batch_size is None:
            batch_size = n_samples

        full_data = {'X': X, 'y': y, 'indices': indices}

        def batched_data_factory():
            perm = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                yield {
                    'X': X[idx],
                    'y': y[idx],
                    'indices': indices[idx],
                }

        # Phase 1: Warmup fit with default regularization
        if verbose:
            print(f"Phase 1: Warmup ({warmup_epochs} epochs) with default σ_eff={self.noise_scale:.2f}")

        warmup_losses, warmup_params = self.fit(
            batched_data_factory=batched_data_factory,
            num_epochs=warmup_epochs,
            learning_rate=learning_rate,
            verbose=verbose,
            **kwargs,
        )

        # Compute predictions and update regularization
        p_hat = np.array(self.predict_proba(full_data, warmup_params))
        self.update_regularization_from_predictions(y, p_hat)

        if verbose:
            print(f"\nPhase 2: Main fit ({num_epochs - warmup_epochs} epochs) with σ_eff={self.noise_scale:.3f}")
            scales = self._get_regularization_scales()
            print(f"  Regularization scales: {list(scales.items())[:3]}...")

        # Phase 2: Main fit with updated regularization
        main_losses, final_params = self.fit(
            batched_data_factory=batched_data_factory,
            initial_values=warmup_params,
            num_epochs=num_epochs - warmup_epochs,
            learning_rate=learning_rate * 0.5,  # Lower LR for fine-tuning
            verbose=verbose,
            **kwargs,
        )

        self.params = final_params
        return warmup_losses + main_losses, final_params

    def _initialize_params(self) -> Dict[str, jax.typing.ArrayLike]:
        """Initialize parameters from decompositions."""
        params = {}

        int_tensors, _, _ = self.intercept_decomp.generate_tensors(dtype=self.dtype)
        for name, tensor in int_tensors.items():
            params[name] = jnp.zeros_like(tensor)

        beta_tensors, _, _ = self.beta_decomp.generate_tensors(dtype=self.dtype)
        for name, tensor in beta_tensors.items():
            params[name] = jnp.zeros_like(tensor)

        return params

    def _get_decompositions(self) -> Dict[str, Decomposed]:
        """Return decompositions for this model."""
        return {
            'intercept': self.intercept_decomp,
            'beta': self.beta_decomp,
        }
