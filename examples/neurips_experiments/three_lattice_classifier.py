"""Three-lattice classifier: intercept + important beta + weak beta."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import jax
import jax.numpy as jnp
import optax
from typing import List, Tuple
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def ordinal_smoothness_penalty(params: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Penalize adjacent differences along ordinal axis."""
    diffs = jnp.diff(params, axis=axis)
    return jnp.sum(diffs ** 2)


def multi_axis_smoothness(params: jnp.ndarray, n_axes: int) -> jnp.ndarray:
    """Apply smoothness penalty along each of the first n_axes."""
    penalty = 0.0
    for axis in range(n_axes):
        if params.shape[axis] > 1:
            penalty += ordinal_smoothness_penalty(params, axis=axis)
    return penalty


class ThreeLatticeClassifier:
    """
    Tiered lattice classifier with selective order expansion.

    Architecture:
    1. Intercept lattice: baseline heterogeneity across latent bins
    2. Top beta: order 2 lattice for top features (highest |β|)
    3. Mid beta: order 1 lattice (main effects) for mid-tier features
    4. Weak beta: global coefficients for rest
    """

    def __init__(
        self,
        n_latents: int = 6,
        bins_per_latent: int = 8,
        intercept_order: int = 2,
        top_beta_order: int = 2,
        mid_beta_order: int = 1,
        top_percentile: float = 95.0,
        mid_percentile: float = 80.0,
        scale_multiplier: float = 2.0,
        smoothness_weight: float = 0.5,
        l1_weight: float = 0.01,
        learning_rate: float = 0.01,
        n_steps: int = 5000,
    ):
        self.n_latents = n_latents
        self.bins_per_latent = bins_per_latent
        self.intercept_order = intercept_order
        self.top_beta_order = top_beta_order
        self.mid_beta_order = mid_beta_order
        self.top_percentile = top_percentile
        self.mid_percentile = mid_percentile
        self.scale_multiplier = scale_multiplier
        self.smoothness_weight = smoothness_weight
        self.l1_weight = l1_weight
        self.learning_rate = learning_rate
        self.n_steps = n_steps

    def _fit_zero_order(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit zero-order (global) logistic regression to get feature importances."""
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr.fit(X, y)
        return lr.coef_[0]

    def _compute_max_order_by_cell_count(self, latent_bins: np.ndarray, min_samples: int = 20, verbose: bool = True):
        """Compute max order based on cell occupancy at each interaction level."""
        N = len(latent_bins)
        k = self.n_latents
        b = self.bins_per_latent

        from itertools import combinations

        max_order = 0
        for order in range(1, k + 1):
            # Check all combinations of 'order' dimensions
            min_count = N
            for dims in combinations(range(k), order):
                # Count samples in each cell for this combination
                cell_counts = {}
                for i in range(N):
                    cell = tuple(latent_bins[i, d] for d in dims)
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1

                if cell_counts:
                    min_count = min(min_count, min(cell_counts.values()))

            if min_count >= min_samples:
                max_order = order
                if verbose:
                    print(f"  Order {order}: min cell count = {min_count} (>= {min_samples}) ✓")
            else:
                if verbose:
                    print(f"  Order {order}: min cell count = {min_count} (< {min_samples}) ✗")
                break

        return max_order

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit the tiered lattice classifier."""
        N = len(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        n_features = X_scaled.shape[1]

        # Step 1: Zero-order fit to select features
        beta_zero = self._fit_zero_order(X_scaled, y)
        abs_beta = np.abs(beta_zero)

        # Three-tier feature selection
        top_threshold = np.percentile(abs_beta, self.top_percentile)
        mid_threshold = np.percentile(abs_beta, self.mid_percentile)

        self.top_features = np.where(abs_beta >= top_threshold)[0]
        self.mid_features = np.where((abs_beta >= mid_threshold) & (abs_beta < top_threshold))[0]
        self.weak_features = np.where(abs_beta < mid_threshold)[0]

        # Select top features for latent dimensions
        importance_order = np.argsort(abs_beta)[::-1]
        self.latent_features = importance_order[:self.n_latents]

        if verbose:
            print(f"N={N}, features={n_features}")
            print(f"Top (p{self.top_percentile}): {len(self.top_features)} features, order {self.top_beta_order}")
            print(f"Mid (p{self.mid_percentile}-{self.top_percentile}): {len(self.mid_features)} features, order {self.mid_beta_order}")
            print(f"Weak: {len(self.weak_features)} features, global")
            print(f"Latent features: {self.latent_features.tolist()}")

        # Compute bin edges for latent features
        latents = X_scaled[:, self.latent_features]
        self.bin_edges = []
        for i in range(self.n_latents):
            edges = np.percentile(latents[:, i],
                                  np.linspace(0, 100, self.bins_per_latent + 1)[1:-1])
            self.bin_edges.append(edges)

        # Compute bin indices
        latent_bins = np.zeros((N, self.n_latents), dtype=int)
        for i in range(self.n_latents):
            latent_bins[:, i] = np.digitize(latents[:, i], self.bin_edges[i])

        # Determine max order based on cell occupancy
        if verbose:
            print("Cell occupancy check:")
        data_max_order = self._compute_max_order_by_cell_count(latent_bins, min_samples=20, verbose=verbose)

        # Use min of requested order and data-supported order
        effective_int_order = min(self.intercept_order, data_max_order)
        effective_top_order = min(self.top_beta_order, data_max_order)
        effective_mid_order = min(self.mid_beta_order, data_max_order)

        # Build dimensions
        dims = [Dimension(f"L{i}", self.bins_per_latent) for i in range(self.n_latents)]
        interactions = Interactions(dimensions=dims)

        # Intercept lattice
        decomp_int = Decomposed(interactions=interactions, param_shape=[1], name="intercept")

        # Top beta lattice (order 2)
        n_top = len(self.top_features)
        decomp_top = Decomposed(interactions=interactions, param_shape=[n_top], name="beta_top")

        # Mid beta lattice (order 1 - main effects only)
        n_mid = len(self.mid_features)
        decomp_mid = Decomposed(interactions=interactions, param_shape=[n_mid], name="beta_mid")

        # Weak beta: global
        n_weak = len(self.weak_features)

        # Get prior scales
        scales_int = decomp_int.generalization_preserving_scales(noise_scale=1.0, total_n=N, c=0.5, per_component=True)
        scales_top = decomp_top.generalization_preserving_scales(noise_scale=1.0, total_n=N, c=0.5, per_component=True)
        scales_mid = decomp_mid.generalization_preserving_scales(noise_scale=1.0, total_n=N, c=0.5, per_component=True)

        # Global beta prior scale
        class_balance = y.mean()
        sigma_eff = 1 / np.sqrt(class_balance * (1 - class_balance))
        tau_weak = sigma_eff / np.sqrt(N)

        # Active components by order
        active_int = [n for n in decomp_int._tensor_parts.keys()
                      if decomp_int.component_order(n) <= effective_int_order]
        active_top = [n for n in decomp_top._tensor_parts.keys()
                      if decomp_top.component_order(n) <= effective_top_order]
        active_mid = [n for n in decomp_mid._tensor_parts.keys()
                      if decomp_mid.component_order(n) <= effective_mid_order]

        if verbose:
            print(f"Intercept: {len(active_int)} components (order <= {effective_int_order})")
            print(f"Top β: {len(active_top)} components × {n_top} features (order <= {effective_top_order})")
            print(f"Mid β: {len(active_mid)} components × {n_mid} features (order <= {effective_mid_order})")
            print(f"Weak β: global ({n_weak} features)")

        # Initialize parameters
        params = {
            "intercept": {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int},
            "beta_top": {n: jnp.zeros(decomp_top._tensor_part_shapes[n]) for n in active_top},
            "beta_mid": {n: jnp.zeros(decomp_mid._tensor_part_shapes[n]) for n in active_mid},
            "beta_weak": jnp.zeros(n_weak),
        }

        # Store for prediction
        self._decomp_int = decomp_int
        self._decomp_top = decomp_top
        self._decomp_mid = decomp_mid
        self._active_int = active_int
        self._active_top = active_top
        self._active_mid = active_mid

        # Convert to JAX arrays
        idx = jnp.array(latent_bins)
        X_top = jnp.array(X_scaled[:, self.top_features])
        X_mid = jnp.array(X_scaled[:, self.mid_features])
        X_weak = jnp.array(X_scaled[:, self.weak_features])
        y_j = jnp.array(y)

        scale_mult = self.scale_multiplier
        smooth_wt = self.smoothness_weight
        l1_wt = self.l1_weight

        def loss_fn(params):
            # Lookup intercept
            int_vals = decomp_int.lookup_flat(idx, params["intercept"])[:, 0]

            # Lookup beta for top features (order 2)
            beta_top = decomp_top.lookup_flat(idx, params["beta_top"])

            # Lookup beta for mid features (order 1)
            beta_mid = decomp_mid.lookup_flat(idx, params["beta_mid"])

            # Weak beta is global
            beta_weak = params["beta_weak"]

            # Compute logits
            logits = (int_vals
                     + jnp.sum(X_top * beta_top, axis=-1)
                     + jnp.sum(X_mid * beta_mid, axis=-1)
                     + jnp.sum(X_weak * beta_weak, axis=-1))
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_j * logits)

            # L2 regularization with theory-based scales
            l2_int = sum(0.5 * jnp.sum(p ** 2) / ((scales_int.get(n, 1.0) * scale_mult) ** 2 + 1e-8)
                         for n, p in params["intercept"].items())
            l2_top = sum(0.5 * jnp.sum(p ** 2) / ((scales_top.get(n, 1.0) * scale_mult) ** 2 + 1e-8)
                         for n, p in params["beta_top"].items())
            l2_mid = sum(0.5 * jnp.sum(p ** 2) / ((scales_mid.get(n, 1.0) * scale_mult) ** 2 + 1e-8)
                         for n, p in params["beta_mid"].items())
            l2_weak = 0.5 * jnp.sum(beta_weak ** 2) / (tau_weak ** 2 + 1e-8)

            # L1 sparsity
            l1_int = sum(jnp.sum(jnp.abs(p)) for p in params["intercept"].values())
            l1_top = sum(jnp.sum(jnp.abs(p)) for p in params["beta_top"].values())
            l1_mid = sum(jnp.sum(jnp.abs(p)) for p in params["beta_mid"].values())
            l1_weak = jnp.sum(jnp.abs(beta_weak))

            # Smoothness penalty
            smooth = 0.0
            for lattice_params, decomp in [(params["intercept"], decomp_int),
                                            (params["beta_top"], decomp_top),
                                            (params["beta_mid"], decomp_mid)]:
                for n, p in lattice_params.items():
                    order = decomp.component_order(n)
                    if order > 0:
                        smooth += multi_axis_smoothness(p, n_axes=order)

            reg = ((l2_int + l2_top + l2_mid + l2_weak) / N
                   + l1_wt * (l1_int + l1_top + l1_mid + l1_weak) / N
                   + smooth_wt * smooth / N)
            return bce + reg

        # Optimizer
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001, peak_value=0.02, warmup_steps=500,
            decay_steps=self.n_steps - 500, end_value=0.001
        )
        opt = optax.adam(learning_rate=schedule)
        opt_state = opt.init(params)

        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for i in range(self.n_steps + 1):
            params, opt_state, loss = step(params, opt_state)
            if verbose and i % 1000 == 0:
                print(f"  Step {i}: loss = {loss:.4f}")

        self.params = params
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X)

        # Compute bin indices
        latents = X_scaled[:, self.latent_features]
        latent_bins = np.zeros((len(X), self.n_latents), dtype=int)
        for i in range(self.n_latents):
            latent_bins[:, i] = np.digitize(latents[:, i], self.bin_edges[i])

        idx = jnp.array(latent_bins)
        X_top = jnp.array(X_scaled[:, self.top_features])
        X_mid = jnp.array(X_scaled[:, self.mid_features])
        X_weak = jnp.array(X_scaled[:, self.weak_features])

        int_vals = self._decomp_int.lookup_flat(idx, self.params["intercept"])[:, 0]
        beta_top = self._decomp_top.lookup_flat(idx, self.params["beta_top"])
        beta_mid = self._decomp_mid.lookup_flat(idx, self.params["beta_mid"])
        beta_weak = self.params["beta_weak"]

        logits = (int_vals
                 + jnp.sum(X_top * beta_top, axis=-1)
                 + jnp.sum(X_mid * beta_mid, axis=-1)
                 + jnp.sum(X_weak * beta_weak, axis=-1))
        probs = 1 / (1 + jnp.exp(-logits))
        return np.array(probs)


def run_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, **kwargs) -> Tuple[float, float, List[float]]:
    """Run proper k-fold CV with per-fold feature selection."""
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\nFold {fold_idx + 1}/{n_splits}:")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = ThreeLatticeClassifier(**kwargs)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, probs)
        aucs.append(auc)
        print(f"  AUC = {auc:.4f}")

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    print(f"\nMean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    return mean_auc, std_auc, aucs


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    print("=" * 60)
    print("BIORESPONSE - Three-Lattice Classifier")
    print("=" * 60)

    data = fetch_openml(name='bioresponse', version=1, as_frame=True, parser='auto')
    X = data.data.values.astype(np.float32)
    y = data.target.astype(float).values

    print(f"Data: N={len(y)}, features={X.shape[1]}, pos_rate={y.mean():.3f}")

    mean_auc, std_auc, aucs = run_cv(
        X, y,
        n_latents=6,
        bins_per_latent=8,
        intercept_order=2,
        top_beta_order=2,
        mid_beta_order=1,
        top_percentile=95.0,
        mid_percentile=80.0,
        scale_multiplier=2.0,
        smoothness_weight=0.5,
        l1_weight=0.01,
        n_steps=5000,
    )

    print(f"\nFinal: {mean_auc:.4f} +/- {std_auc:.4f}")
