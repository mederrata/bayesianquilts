"""Unified lattice classifier with ordinal smoothness regularization."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import jax
import jax.numpy as jnp
import optax
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
from bayesianquilts.jax.parameter import Decomposed, Interactions, Dimension


def within_bin_normalize(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Transform continuous values to [0,1] based on position within their bin."""
    values = np.asarray(values, dtype=float)
    full_edges = np.concatenate([[values.min()], bin_edges, [values.max()]])
    bins = np.digitize(values, bin_edges)

    lower = full_edges[bins]
    upper = full_edges[bins + 1]
    width = np.where(upper - lower == 0, 1.0, upper - lower)

    return np.clip((values - lower) / width, 0, 1)


def ordinal_smoothness_penalty(params: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Compute smoothness penalty for ordinal dimension (penalize adjacent differences)."""
    diffs = jnp.diff(params, axis=axis)
    return jnp.sum(diffs ** 2)


def multi_axis_smoothness_penalty(params: jnp.ndarray, n_ordinal_axes: int) -> jnp.ndarray:
    """Apply smoothness penalty along each of the first n_ordinal_axes axes.

    For interaction terms, penalizes differences along each ordinal dimension.
    E.g., for shape (5, 6, n_features), penalizes along axis 0 and axis 1.
    """
    penalty = 0.0
    for axis in range(n_ordinal_axes):
        if params.shape[axis] > 1:  # Need at least 2 points to compute diff
            penalty += ordinal_smoothness_penalty(params, axis=axis)
    return penalty


class LatticeClassifier:
    """
    Lattice-based classifier with ordinal smoothness regularization.

    Uses two-lattice architecture:
    - Intercept lattice: captures baseline heterogeneity
    - Beta lattice: captures coefficient variation

    Supports:
    - Automatic latent extraction via PCA for high-dim data
    - Ordinal smoothness for binned continuous variables
    - Within-bin normalization for piecewise linear effects
    """

    def __init__(
        self,
        n_latents: int = 3,
        bins_per_latent: int = 5,
        intercept_order: int = 3,
        beta_order: int = 2,
        scale_multiplier: float = 50.0,
        smoothness_weight: float = 1.0,
        l1_weight: float = 0.0,
        l2_weight: float = 1.0,
        learning_rate: float = 0.01,
        n_steps: int = 5000,
        fixed_features: np.ndarray = None,
    ):
        """
        Args:
            n_latents: Number of PCA components for latent dimensions
            bins_per_latent: Number of bins per latent dimension
            intercept_order: Max interaction order for intercept lattice
            beta_order: Max interaction order for beta lattice
            scale_multiplier: Multiplier for prior scales (higher = more relaxed)
            smoothness_weight: Weight for ordinal smoothness penalty
            l1_weight: Weight for L1 (sparsity) penalty on beta coefficients
            l2_weight: Weight for L2 (ridge) penalty (default 1.0)
            learning_rate: Adam learning rate
            n_steps: Number of optimization steps
            fixed_features: Pre-selected feature indices (if None, selects via LR)
        """
        self.n_latents = n_latents
        self.fixed_features = fixed_features
        self.bins_per_latent = bins_per_latent
        self.intercept_order = intercept_order
        self.beta_order = beta_order
        self.scale_multiplier = scale_multiplier
        self.smoothness_weight = smoothness_weight
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.learning_rate = learning_rate
        self.n_steps = n_steps

        self.pca = None
        self.scaler = None
        self.bin_edges = None
        self.params = None

    def _select_top_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select top features using LR coefficient magnitudes."""
        lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
        lr.fit(X, y)
        importance = np.abs(lr.coef_[0])
        return np.argsort(importance)[::-1][:self.n_latents]

    def _extract_latents(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract top features via LR selection, return bins and normalized values."""
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Use fixed features if provided, otherwise select via LR
            if self.fixed_features is not None:
                self.top_features = self.fixed_features
            else:
                self.top_features = self._select_top_features(X_scaled, y)
            latents = X_scaled[:, self.top_features]

            # Compute bin edges for each feature (quantile-based)
            self.bin_edges = []
            for i in range(self.n_latents):
                edges = np.percentile(latents[:, i],
                                     np.linspace(0, 100, self.bins_per_latent + 1)[1:-1])
                self.bin_edges.append(edges)
        else:
            X_scaled = self.scaler.transform(X)
            latents = X_scaled[:, self.top_features]

        # Compute bins and normalized values
        latent_bins = np.zeros((len(latents), self.n_latents), dtype=int)
        latent_norms = np.zeros((len(latents), self.n_latents))

        for i in range(self.n_latents):
            latent_bins[:, i] = np.digitize(latents[:, i], self.bin_edges[i])
            latent_norms[:, i] = within_bin_normalize(latents[:, i], self.bin_edges[i])

        return latents, latent_bins, latent_norms


    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit the lattice classifier."""
        N = len(y)

        # Extract top features via LR selection
        latents, latent_bins, latent_norms = self._extract_latents(X, y, fit=True)

        # Use scaled features directly (simpler, like the working config)
        X_scaled = self.scaler.transform(X)
        n_features = X_scaled.shape[1]

        if verbose:
            print(f"N={N}, features={n_features}, top_features={self.top_features.tolist()}")

        # Build lattice dimensions (single intercept lattice, global beta)
        dims_int = [Dimension(f"F{self.top_features[i]}", self.bins_per_latent)
                    for i in range(self.n_latents)]

        interactions_int = Interactions(dimensions=dims_int)
        decomp_int = Decomposed(interactions=interactions_int, param_shape=[1], name="intercept")

        # Theory-based prior scales
        prior_scales_int = decomp_int.generalization_preserving_scales(
            noise_scale=1.0, total_n=N, c=0.5, per_component=True
        )

        # Global beta prior scale (match improve_bioresponse_best.py)
        class_balance = y.mean()
        sigma_eff = 1 / np.sqrt(class_balance * (1 - class_balance))
        c = 0.5
        bound_factor = np.sqrt(c / (1 - c))  # = 1.0
        tau_beta = bound_factor * sigma_eff / np.sqrt(N)

        # Build indices
        idx_int = jnp.array(latent_bins)

        X_j = jnp.array(X_scaled)
        y_j = jnp.array(y)

        # Initialize parameters (intercept lattice + global beta)
        active_int = [n for n in decomp_int._tensor_parts.keys()
                      if decomp_int.component_order(n) <= self.intercept_order]

        params = {
            "intercept": {n: jnp.zeros(decomp_int._tensor_part_shapes[n]) for n in active_int},
            "beta": jnp.zeros(n_features),
        }

        if verbose:
            total_int_cells = np.prod([d.cardinality for d in dims_int])
            print(f"Intercept: {total_int_cells} cells, {len(active_int)} components")
            print(f"Beta: global vector ({n_features} features), tau={tau_beta:.4f}")

        # Store for prediction
        self._decomp_int = decomp_int
        self._active_int = active_int
        self._dims_int = dims_int
        self._tau_beta = tau_beta

        scale_mult = self.scale_multiplier
        smooth_weight = self.smoothness_weight
        l1_wt = self.l1_weight
        l2_wt = self.l2_weight

        def loss_fn(params):
            int_vals = decomp_int.lookup_flat(idx_int, params["intercept"])
            logits = jnp.sum(X_j * params["beta"], axis=-1) + int_vals[:, 0]
            bce = jnp.mean(jnp.logaddexp(0, logits) - y_j * logits)

            # L2 regularization on intercept with theory-based scales
            l2_int = sum(0.5 * jnp.sum(p ** 2) / ((prior_scales_int.get(n, 1.0) * scale_mult) ** 2 + 1e-8)
                        for n, p in params["intercept"].items())

            # L2 regularization on global beta
            l2_beta = 0.5 * jnp.sum(params["beta"] ** 2) / (tau_beta ** 2 + 1e-8)

            # L1 regularization for sparsity on beta
            l1_beta = jnp.sum(jnp.abs(params["beta"]))

            # Ordinal smoothness along each ordinal axis
            smooth_penalty = 0.0
            for n, p in params["intercept"].items():
                order = decomp_int.component_order(n)
                if order > 0:  # Skip global intercept (order 0)
                    smooth_penalty += multi_axis_smoothness_penalty(p, n_ordinal_axes=order)

            reg = l2_wt * (l2_int + l2_beta) / N + l1_wt * l1_beta / N + smooth_weight * smooth_penalty / N
            return bce + reg

        # Learning rate schedule (warmup + cosine decay like working config)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.001,
            peak_value=0.02,
            warmup_steps=500,
            decay_steps=self.n_steps - 500,
            end_value=0.001,
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
        _, latent_bins, _ = self._extract_latents(X, y=None, fit=False)

        X_scaled = self.scaler.transform(X)

        idx_int = jnp.array(latent_bins)
        X_j = jnp.array(X_scaled)

        int_vals = self._decomp_int.lookup_flat(idx_int, self.params["intercept"])
        logits = jnp.sum(X_j * self.params["beta"], axis=-1) + int_vals[:, 0]

        probs = 1 / (1 + jnp.exp(-logits))
        return np.array(probs)


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    fixed_features: np.ndarray = None,
    **kwargs
) -> Tuple[float, float, List[float]]:
    """Run cross-validation with LatticeClassifier.

    Args:
        fixed_features: Pre-selected feature indices to use for all folds.
                       If None, selects within each fold's training data (proper CV).
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\nFold {fold_idx + 1}/{n_splits}:")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Feature selection within training fold only (no leakage)
        fold_features = fixed_features
        if fold_features is None:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            lr = LogisticRegression(max_iter=2000, C=0.1, random_state=42)
            lr.fit(X_train_scaled, y_train)
            importance = np.abs(lr.coef_[0])
            n_latents = kwargs.get('n_latents', 4)
            fold_features = np.argsort(importance)[::-1][:n_latents]

        clf = LatticeClassifier(fixed_features=fold_features, **kwargs)
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
    print("BIORESPONSE - Lattice with Ordinal Smoothness")
    print("=" * 60)

    data = fetch_openml(name='bioresponse', version=1, as_frame=True, parser='auto')
    X = data.data.values.astype(np.float32)
    y = data.target.astype(float).values

    print(f"Data: N={len(y)}, features={X.shape[1]}, pos_rate={y.mean():.3f}")

    # Config matching improve_bioresponse_best.py (0.8436 AUC)
    # 4 top LR features × 8 bins = 4096 cells, order 2
    mean_auc, std_auc, aucs = run_cv(
        X, y,
        n_latents=4,         # Top 4 features by LR importance
        bins_per_latent=8,   # 8 bins per feature
        intercept_order=2,   # Main + pairwise only
        beta_order=2,        # Not used (global beta)
        scale_multiplier=2.0,  # Match theory-based priors
        smoothness_weight=0.0, # No smoothness for bioresponse
        l1_weight=0.0,       # No L1
        l2_weight=1.0,
        learning_rate=0.01,
        n_steps=5000,
    )

    print(f"\nFinal: {mean_auc:.4f} +/- {std_auc:.4f}")
